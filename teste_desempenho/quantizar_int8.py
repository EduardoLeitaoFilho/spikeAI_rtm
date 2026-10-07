"""
Quantiza camadas do RTMPose-x para INT8 (NNCF / OpenVINO) para rodar na NPU.

O RTMPose-x em FP16 não compila na NPU Intel AI Boost: a camada Conv_323
precisa de ~2,3 MB de memória interna e só há ~1,9 MB. Em INT8 os pesos caem
pela metade e a camada passa a caber.

Por padrão, SÓ a Conv_323 é quantizada e o resto do modelo fica em ponto
flutuante (FP16 na NPU). Não quantize o modelo inteiro para a NPU: veja
teste_desempenho/resultados_arquivo/HISTORICO_NPU.md.

A calibração usa recortes de pessoas do próprio vídeo do projeto, preparados
exatamente como a rtmlib prepara a entrada do RTMPose (recorte + redimensiona +
normaliza). Assim a quantização se ajusta aos dados reais.

Uso (a partir da raiz do projeto, no ambiente .venv-quant):
    python teste_desempenho/quantizar_int8.py                      # só a Conv_323
    python teste_desempenho/quantizar_int8.py --camadas Conv_323/WithoutBiases MatMul_339
    python teste_desempenho/quantizar_int8.py --camadas todas --saida teste_desempenho/modelos/rtmpose-x_int8_total.xml   # para a CPU

Saída: teste_desempenho/modelos/rtmpose-x_int8.xml (+ .bin), usado por
configs/x_npu.yaml.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_config  # noqa: E402

# Config de onde vêm o modelo FP32, o detector e o vídeo (a referência FP32)
SOURCE_CONFIG = BENCH_DIR / "configs" / "x_cpu.yaml"
OUTPUT_MODEL = BENCH_DIR / "modelos" / "rtmpose-x_int8.xml"

# Camada que não cabe na memória interna da NPU em FP16 (nome do nó no OpenVINO)
DEFAULT_LAYERS = ["Conv_323/WithoutBiases"]


def collect_calibration_inputs(cfg, n_frames: int, max_samples: int, seed: int = 0):
    """Recortes de pessoas de `n_frames` frames espalhados pelo vídeo, já no formato do RTMPose."""
    from rtmlib import RTMPose, YOLOX
    from rtmlib.tools.file import download_checkpoint

    pose_cfg = cfg.pose
    det_cfg = pose_cfg.detector
    rt = det_cfg.runtime
    det = YOLOX(download_checkpoint(det_cfg.checkpoint) if det_cfg.checkpoint.startswith("http") else det_cfg.checkpoint,
                model_input_size=det_cfg.input_size.as_tuple(), backend=rt.backend, device=rt.device)
    # Só o preprocess do RTMPose é usado aqui (mesma normalização da inferência)
    pose_onnx = download_checkpoint(pose_cfg.checkpoint) if pose_cfg.checkpoint.startswith("http") else pose_cfg.checkpoint
    pose = RTMPose(pose_onnx, model_input_size=pose_cfg.input_size.as_tuple(), backend="onnxruntime", device="cpu")

    cap = cv2.VideoCapture(cfg.input.source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = set(np.linspace(0, total - 1, num=min(n_frames, total), dtype=int).tolist())

    # Leitura sequencial: pular direto para um frame (CAP_PROP_POS_FRAMES) às
    # vezes devolve um frame corrompido e derruba o OpenCV
    samples = []
    frame_id = done = 0
    while done < len(frame_ids):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id in frame_ids:
            done += 1
            for bbox in det(frame):
                img, _, _ = pose.preprocess(frame, bbox)
                samples.append(np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32))
            print(f"  frame {frame_id:>4} ({done}/{len(frame_ids)}): {len(samples)} recortes", flush=True)
        frame_id += 1
    cap.release()

    if len(samples) > max_samples:
        rng = np.random.default_rng(seed)
        samples = [samples[j] for j in sorted(rng.choice(len(samples), max_samples, replace=False))]
    return samples, pose_onnx


def main():
    parser = argparse.ArgumentParser(description="Quantiza o RTMPose-x para INT8 (NPU)")
    parser.add_argument("--frames", type=int, default=30, help="Frames do vídeo usados na calibração")
    parser.add_argument("--amostras", type=int, default=300, help="Máximo de recortes de pessoas na calibração")
    parser.add_argument("--camadas", nargs="+", default=DEFAULT_LAYERS,
                        help="Nós a quantizar (padrão: %(default)s), ou 'todas' para o modelo inteiro")
    parser.add_argument("--saida", type=Path, default=OUTPUT_MODEL,
                        help="Modelo INT8 de saída (padrão: %(default)s)")
    args = parser.parse_args()
    output_model = args.saida.resolve()

    try:
        import nncf
        import openvino as ov
    except ImportError:
        sys.exit("Requer nncf e openvino: pip install -r requirements.txt")

    cfg = load_config(SOURCE_CONFIG, base_dir=PROJECT_ROOT)
    print(f"Coletando dados de calibração de {cfg.input.source} ...")
    samples, pose_onnx = collect_calibration_inputs(cfg, args.frames, args.amostras)
    print(f"{len(samples)} recortes de calibração")

    model = ov.Core().read_model(pose_onnx)
    size = cfg.pose.input_size
    model.reshape([1, 3, size.height, size.width])  # a NPU exige formato fixo

    # Quantiza só as camadas pedidas: todas as outras vão para o ignored_scope
    ignored_scope = None
    if args.camadas != ["todas"]:
        names = [op.get_friendly_name() for op in model.get_ordered_ops()]
        missing = [c for c in args.camadas if c not in names]
        if missing:
            sys.exit(f"Camada(s) inexistente(s) no modelo: {missing}")
        ignored_scope = nncf.IgnoredScope(names=[n for n in names if n not in args.camadas],
                                          validate=False)
    print(f"Quantizando (NNCF): {'modelo inteiro' if ignored_scope is None else args.camadas} ...")
    start = time.perf_counter()
    quantized = nncf.quantize(
        model,
        nncf.Dataset(samples),
        # MIXED: pesos simétricos, ativações assimétricas; recomendado para
        # ativações não-ReLU (o RTMPose usa SiLU)
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=len(samples),
        ignored_scope=ignored_scope,
    )
    print(f"Quantização concluída em {time.perf_counter() - start:.1f} s")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, str(output_model), compress_to_fp16=False)
    fp32_mb = Path(pose_onnx).stat().st_size / 2**20
    int8_mb = output_model.with_suffix(".bin").stat().st_size / 2**20
    print(f"Modelo INT8 salvo em: {output_model}")
    print(f"Tamanho dos pesos: {fp32_mb:.1f} MB (FP32) -> {int8_mb:.1f} MB (quantizado)")


if __name__ == "__main__":
    main()
