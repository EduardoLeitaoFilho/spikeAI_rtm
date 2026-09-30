"""
Roda o pipeline de pose em cada dispositivo configurado e compara o desempenho.

Uso (a partir da raiz do projeto):
    python teste_desempenho/run_benchmark.py                 # todas as configs
    python teste_desempenho/run_benchmark.py cpu npu         # só as escolhidas
    python teste_desempenho/run_benchmark.py --verificar     # só checa o ambiente
    python teste_desempenho/run_benchmark.py --baixar-modelos  # só baixa os modelos
    python teste_desempenho/run_benchmark.py --so-relatorio    # só remonta o comparativo

Cada config em teste_desempenho/configs/<nome>.yaml roda em um processo
separado (python src/main.py --config ...), para que um dispositivo não
interfira no outro. Dispositivos indisponíveis nesta máquina são pulados.

Saídas:
    teste_desempenho/resultados/<nome>/run_info.json   (detalhes de cada execução)
    teste_desempenho/resultados/comparativo.csv        (uma linha por dispositivo)
    teste_desempenho/resultados/comparativo.md         (tabela para relatório)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
CONFIGS_DIR = BENCH_DIR / "configs"
RESULTS_DIR = BENCH_DIR / "resultados"
MAIN_SCRIPT = PROJECT_ROOT / "src" / "main.py"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config import ConfigError, load_config  # noqa: E402
from comparar_precisao import REFERENCE, precision_columns  # noqa: E402
from runtime_check import (  # noqa: E402
    DeviceUnavailable, check_runtime, describe_system, device_name, prepare_checkpoint,
    runtime_label,
)

# (coluna, caminho dentro do run_info.json)
COLUMNS = [
    ("modelo_pose", ("models", "pose", "name")),
    ("detector", ("models", "detector", "name")),
    ("execucao", ("runtime", "label")),
    ("precisao_det", ("runtime", "detector", "precision")),
    ("precisao_pose", ("runtime", "pose", "precision")),
    ("dispositivo", ("runtime", "device_name")),
    ("frames", ("video", "frames_processed")),
    ("carga_modelos_s", ("performance", "model_load_s")),
    ("detector_ms", ("performance", "inference_ms", "detector", "mean")),
    ("pose_ms", ("performance", "inference_ms", "pose", "mean")),
    ("inferencia_ms", ("performance", "inference_ms", "total", "mean")),
    ("inferencia_p95_ms", ("performance", "inference_ms", "total", "p95")),
    ("aquecimento_ms", ("performance", "inference_ms", "total", "warmup_mean")),
    ("fps_inferencia", ("performance", "inference_fps")),
    ("fps_pipeline", ("performance", "pipeline_fps")),
    ("tempo_total_s", ("performance", "processing_wall_s")),
    ("taxa_deteccao_pct", ("detection", "detection_rate_pct")),
    ("confianca_media", ("detection", "mean_keypoint_confidence")),
    ("observacao", ("error",)),
]

# Código de saída de src/main.py quando o modelo/dispositivo não pode ser usado
EXIT_MODEL_UNAVAILABLE = 3


def _get(data, path):
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def _order(name):
    """Ordem da tabela: RTMPose-x, RTMPose-m, híbridos; dentro de cada grupo, por dispositivo."""
    groups = ("x_", "m_", "hibrido")
    devices = ("cpu", "gpu_intel", "npu", "gpu_cuda")
    group = next((i for i, g in enumerate(groups) if name.startswith(g)), len(groups))
    device = next((i for i, d in enumerate(devices) if name.endswith(d)), len(devices))
    return group, device, name


def available_configs():
    return sorted((p.stem for p in CONFIGS_DIR.glob("*.yaml")), key=_order)


def check(name):
    """Retorna (cfg, None) se o dispositivo está disponível, ou (cfg|None, motivo)."""
    path = CONFIGS_DIR / f"{name}.yaml"
    try:
        cfg = load_config(path, base_dir=PROJECT_ROOT)
    except ConfigError as e:
        hint = " (gere o modelo antes: python teste_desempenho/quantizar_int8.py)" if "int8" in str(e) else ""
        return None, f"config inválida: {e}{hint}"
    try:
        check_runtime(cfg.pose.detector.runtime)
        check_runtime(cfg.pose.runtime)
    except DeviceUnavailable as e:
        return cfg, str(e)
    return cfg, None


def _devices(cfg):
    det, pose = cfg.pose.detector.runtime, cfg.pose.runtime
    if det == pose:
        return device_name(pose)
    return f"det: {device_name(det)} + pose: {device_name(pose)}"


def verify_environment(names):
    system = describe_system()
    print(f"Python {system['python']} | {system['os']}")
    print(f"CPU: {system['cpu']} ({system['cpu_logical_cores']} núcleos lógicos)")
    print("Bibliotecas:", ", ".join(f"{k} {v}" for k, v in system["libraries"].items()))
    print()
    ok = True
    for name in names:
        cfg, reason = check(name)
        if reason is None:
            print(f"  [OK]    {name:<20} -> {_devices(cfg)}")
        else:
            ok = False
            print(f"  [FALTA] {name:<20} -> {reason}")
    return ok


def download_models(names):
    """
    Baixa os checkpoints de todas as configs para o cache da rtmlib, para que o
    download não seja contado como 'carga dos modelos' na primeira execução.
    """
    from rtmlib.tools.file import download_checkpoint

    urls = set()
    configs = [load_config(CONFIGS_DIR / f"{n}.yaml", base_dir=PROJECT_ROOT) for n in names]
    for cfg in configs:
        for ckpt in (cfg.pose.checkpoint, cfg.pose.detector.checkpoint):
            if ckpt.startswith(("http://", "https://")):
                urls.add(ckpt)
    for url in sorted(urls):
        print(f"Modelo em cache: {download_checkpoint(url)}")

    # Versões de formato fixo para a NPU (só onde o dispositivo existe)
    for cfg in configs:
        det = cfg.pose.detector
        for ckpt, size, runtime in ((det.checkpoint, det.input_size, det.runtime),
                                    (cfg.pose.checkpoint, cfg.pose.input_size, cfg.pose.runtime)):
            try:
                check_runtime(runtime)
            except DeviceUnavailable:
                continue
            prepare_checkpoint(ckpt, size, runtime)


def run_one(name, cfg):
    label = runtime_label(cfg.pose.runtime, cfg.pose.detector.runtime)
    print(f"\n{'=' * 70}\n>> {name}: {cfg.pose.model} | {label}\n{'=' * 70}", flush=True)
    run_info_path = cfg.pose.output.run_info
    if run_info_path.exists():
        run_info_path.unlink()  # evita reaproveitar o resultado de uma execução anterior

    proc = subprocess.run(
        [sys.executable, str(MAIN_SCRIPT), "--config", str(CONFIGS_DIR / f"{name}.yaml")],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if proc.returncode == EXIT_MODEL_UNAVAILABLE and run_info_path.exists():
        return row_from_run_info(name, run_info_path)  # status + motivo gravados pelo main.py
    if proc.returncode != 0:
        return {"config": name, "status": f"erro (código {proc.returncode})"}
    if not run_info_path.exists():
        return {"config": name, "status": "erro (run_info.json não gerado)"}
    return row_from_run_info(name, run_info_path)


def row_from_run_info(name, run_info_path, status_suffix=""):
    with open(run_info_path, encoding="utf-8") as f:
        info = json.load(f)
    row = {"config": name, "status": info.get("status", "?") + status_suffix}
    row.update({col: _get(info, path) for col, path in COLUMNS})
    row["cpu_maquina"] = _get(info, ("system", "cpu"))
    row.update(per_person_columns(info, Path(run_info_path).parent / "landmarks.csv"))
    # Observação: motivo da falha e/ou nota da config (ex.: "rodado em INT8")
    row["observacao"] = " | ".join(t for t in (info.get("error"), info.get("note")) if t) or None
    return row


def per_person_columns(info, landmarks_csv):
    """
    Métricas equivalentes entre testes. O RTMPose roda uma vez por pessoa
    detectada, então pose_ms cresce com o número de pessoas no frame; aqui o
    tempo de pose é dividido pelo total de pessoas (lido do landmarks.csv, que
    tem uma linha por pessoa e por frame), com o mesmo aquecimento descartado.
    """
    pose = _get(info, ("performance", "inference_ms", "pose")) or {}
    warmup = _get(info, ("performance", "warmup_frames_excluded")) or 0
    # run_info.json antigos não têm pose.frames: vem de frames_processed - aquecimento
    frames = pose.get("frames") or ((_get(info, ("video", "frames_processed")) or 0) - warmup)
    if not pose.get("mean") or frames <= 0 or not landmarks_csv.exists():
        return {}
    with open(landmarks_csv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        frame_col = next(reader).index("frame")
        people = sum(1 for r in reader if int(r[frame_col]) >= warmup)
    if people == 0:
        return {}
    people_per_frame = people / frames
    pose_per_person = pose["mean"] / people_per_frame
    one_person = _get(info, ("performance", "inference_ms", "detector", "mean"))
    one_person = one_person + pose_per_person if one_person is not None else None
    return {
        "pessoas_por_frame": round(people_per_frame, 2),
        "pose_ms_por_pessoa": round(pose_per_person, 2),
        "inferencia_1_pessoa_ms": round(one_person, 2) if one_person is not None else None,
        "fps_1_pessoa": round(1000 / one_person, 2) if one_person else None,
    }


def previous_row(name):
    """Resultado de uma execução anterior (para o comparativo não perder dispositivos)."""
    try:
        run_info = load_config(CONFIGS_DIR / f"{name}.yaml", base_dir=PROJECT_ROOT).pose.output.run_info
    except ConfigError:
        # Config inválida nesta máquina (ex.: modelo INT8 do x_npu não gerado aqui):
        # o resultado salvo em outra máquina continua valendo
        run_info = RESULTS_DIR / name / "run_info.json"
    if not run_info.exists():
        return None
    return row_from_run_info(name, run_info, " (execução anterior)")


PER_PERSON_COLUMNS = ["pessoas_por_frame", "pose_ms_por_pessoa", "inferencia_1_pessoa_ms",
                      "fps_1_pessoa"]

PRECISION_COLUMNS = ["pessoas_pareadas_pct", "erro_medio_px", "nme_pct", "pck_5pct",
                     "delta_confianca", "eventos"]


def add_precision(rows):
    """Concordância de cada teste com a referência FP32 (ver comparar_precisao.py)."""
    for row in rows:
        if str(row.get("status", "")).startswith("completed"):
            row.update(precision_columns(row["config"]))


def write_reports(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    add_precision(rows)
    fields = (["config", "status", "cpu_maquina"] + [c for c, _ in COLUMNS]
              + PER_PERSON_COLUMNS + PRECISION_COLUMNS)

    csv_path = RESULTS_DIR / "comparativo.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    md_fields = ["config", "status", "modelo_pose", "execucao", "cpu_maquina", "precisao_det",
                 "precisao_pose", "carga_modelos_s", "detector_ms",
                 "pose_ms", "inferencia_ms", "inferencia_p95_ms", "fps_inferencia", "fps_pipeline",
                 "pessoas_por_frame", "pose_ms_por_pessoa", "inferencia_1_pessoa_ms", "fps_1_pessoa",
                 "taxa_deteccao_pct", "confianca_media", "nme_pct", "pck_5pct", "eventos",
                 "observacao"]
    fmt = lambda v: "" if v is None else str(v).replace("|", "/")  # noqa: E731
    lines = [
        f"# Comparativo de desempenho ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        "Tempos de inferência em ms por frame (média após o aquecimento).",
        f"Precisão (nme_pct, pck_5pct, eventos): concordância com a referência FP32 `{REFERENCE}`; "
        "nme_pct = erro médio em % do tamanho da pessoa (menor = melhor), "
        "pck_5pct = % de keypoints a menos de 5% do tamanho da pessoa (maior = melhor).",
        "",
        "Métricas equivalentes entre testes: o RTMPose roda uma vez por pessoa detectada, então "
        "pose_ms depende de quantas pessoas há no frame. pose_ms_por_pessoa = pose_ms / "
        "pessoas_por_frame; inferencia_1_pessoa_ms = detector_ms + pose_ms_por_pessoa "
        "(estimativa para um frame com uma pessoa) e fps_1_pessoa = 1000 / inferencia_1_pessoa_ms. "
        "cpu_maquina identifica a máquina de cada teste.",
        "",
        "| " + " | ".join(md_fields) + " |",
        "|" + "---|" * len(md_fields),
    ]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in md_fields) + " |" for r in rows]
    md_path = RESULTS_DIR / "comparativo.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines[4:]))
    print(f"\nComparativo salvo em:\n  {csv_path}\n  {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Teste de desempenho do pipeline de pose por dispositivo")
    parser.add_argument("configs", nargs="*",
                        help=f"Configs a rodar (padrão: todas). Disponíveis: {available_configs()}")
    parser.add_argument("--verificar", action="store_true",
                        help="Só verifica quais dispositivos estão disponíveis, sem rodar")
    parser.add_argument("--baixar-modelos", action="store_true",
                        help="Só baixa os modelos para o cache local, sem rodar")
    parser.add_argument("--so-relatorio", action="store_true",
                        help="Só remonta o comparativo a partir dos run_info.json salvos, sem rodar")
    args = parser.parse_args()

    if args.so_relatorio:
        rows = []
        for name in available_configs():
            row = previous_row(name)
            if row is not None:
                row["status"] = row["status"].replace(" (execução anterior)", "")
            else:  # nunca rodou: mostra o motivo (ex.: CUDA sem GPU NVIDIA)
                _, reason = check(name)
                row = {"config": name, "status": "indisponível" if reason else "não executado",
                       "observacao": reason}
            rows.append(row)
        write_reports(rows)
        return

    names = args.configs or available_configs()
    unknown = [n for n in names if n not in available_configs()]
    if unknown:
        parser.error(f"config(s) inexistente(s): {unknown}. Disponíveis: {available_configs()}")

    if args.baixar_modelos:
        download_models(names)
        return

    if args.verificar:
        sys.exit(0 if verify_environment(names) else 1)

    current = {}
    for name in names:
        cfg, reason = check(name)
        if reason is not None:
            print(f"\n>> {name}: PULADO - {reason}")
            current[name] = {"config": name, "status": "indisponível", "observacao": reason}
            continue
        current[name] = run_one(name, cfg)

    # O comparativo sempre lista todas as configs: as não rodadas agora
    # entram com o último resultado salvo, se houver.
    rows = []
    for name in available_configs():
        row = current.get(name) or previous_row(name)
        if row is not None:
            rows.append(row)

    write_reports(rows)


if __name__ == "__main__":
    main()
