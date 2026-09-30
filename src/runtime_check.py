"""
Verificação e identificação do dispositivo de inferência (CPU, GPU, NPU).

- check_runtime(): antes de carregar os modelos, confirma que o backend e o
  dispositivo pedidos no YAML existem nesta máquina.
- verify_active(): depois de carregar, confirma que o dispositivo está mesmo
  em uso. O onnxruntime volta para CPU silenciosamente se o CUDA falhar, o que
  invalidaria qualquer comparação de desempenho.
- describe_runtime(): nome do hardware e versões das bibliotecas (run_info.json).
"""

import os
import platform
import subprocess
import sys
from importlib import metadata

from config import RuntimeConfig


class DeviceUnavailable(RuntimeError):
    """O dispositivo configurado não está disponível nesta máquina."""


def _is_cuda(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


def _cuda_index(device: str) -> int:
    return int(device.split(":")[1]) if ":" in device else 0


def _openvino_devices():
    try:
        from openvino import Core
    except ImportError as e:
        raise DeviceUnavailable(
            "Backend 'openvino' requer o pacote openvino (pip install -r requirements.txt)"
        ) from e
    return Core()


def check_runtime(runtime: RuntimeConfig):
    """Falha com DeviceUnavailable se o backend/dispositivo não puder ser usado."""
    if runtime.backend == "onnxruntime":
        import onnxruntime as ort

        if _is_cuda(runtime.device):
            # Carrega as DLLs de CUDA/cuDNN instaladas via pip (onnxruntime-gpu[cuda,cudnn]),
            # dispensando a instalação manual do CUDA Toolkit.
            if hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls()
                except Exception as e:  # sem pacotes nvidia-* o erro real aparece abaixo
                    print(f"Aviso: preload_dlls falhou: {e}")
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise DeviceUnavailable(
                    "CUDA indisponível: onnxruntime sem CUDAExecutionProvider. "
                    "Nesta máquina é preciso GPU NVIDIA + driver e o ambiente instalado com "
                    "'setup.ps1 -Alvo cuda' (requirements-cuda.txt). "
                    f"Providers disponíveis: {ort.get_available_providers()}"
                )

    elif runtime.backend == "openvino":
        core = _openvino_devices()
        wanted = runtime.device.upper()
        if not any(d.split(".")[0] == wanted for d in core.available_devices):
            raise DeviceUnavailable(
                f"OpenVINO não encontrou o dispositivo '{wanted}'. "
                f"Disponíveis: {core.available_devices}"
            )


def prepare_checkpoint(checkpoint: str, input_size, runtime: RuntimeConfig) -> str:
    """
    Ajusta o checkpoint ao dispositivo, quando necessário.

    A NPU (OpenVINO) só compila modelos com formato de entrada fixo, e os ONNX
    da rtmlib têm o batch dinâmico (o processo cai com erro nativo, sem
    exceção Python). Para 'npu', o modelo é convertido uma única vez para
    OpenVINO IR (.xml/.bin) com entrada [1, 3, altura, largura] e reaproveitado
    do cache nas execuções seguintes.
    """
    if runtime.backend != "openvino" or runtime.device != "npu":
        return checkpoint

    import openvino as ov
    from rtmlib.tools.file import download_checkpoint

    onnx_path = checkpoint if os.path.exists(checkpoint) else download_checkpoint(checkpoint)
    static_path = os.path.splitext(onnx_path)[0] + "_static_b1.xml"
    if os.path.exists(static_path):
        return static_path

    model = ov.Core().read_model(onnx_path)
    shape = model.input(0).get_partial_shape()
    if shape.is_static:
        return onnx_path

    # NCHW: batch = 1; canais/altura/largura do próprio modelo quando fixos, senão do YAML
    fallback = [1, 3, input_size.height, input_size.width]
    static = [1] + [d.get_length() if d.is_static else fallback[i]
                    for i, d in enumerate(shape) if i > 0]
    model.reshape(static)
    ov.save_model(model, static_path, compress_to_fp16=False)
    print(f"Modelo convertido para formato fixo {static} (NPU): {static_path}")
    return static_path


def verify_active(tool, runtime: RuntimeConfig):
    """Confirma que a sessão do onnxruntime de um modelo está no provider pedido."""
    if runtime.backend != "onnxruntime" or not _is_cuda(runtime.device):
        return
    active = tool.session.get_providers()
    if active[0] != "CUDAExecutionProvider":
        raise DeviceUnavailable(
            f"CUDA foi pedido, mas o onnxruntime carregou {active[0]} "
            "(fallback silencioso para CPU). Verifique driver NVIDIA e pacotes nvidia-*."
        )


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or "desconhecido"


def _nvidia_gpu_name(index: int) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        name, driver = [p.strip() for p in out.split(",")]
        return f"{name} (driver {driver})"
    except (OSError, subprocess.SubprocessError, ValueError):
        return "GPU NVIDIA (nvidia-smi indisponível)"


def device_name(runtime: RuntimeConfig) -> str:
    if runtime.backend == "openvino":
        core = _openvino_devices()
        return core.get_property(runtime.device.upper(), "FULL_DEVICE_NAME")
    if _is_cuda(runtime.device):
        return _nvidia_gpu_name(_cuda_index(runtime.device))
    if runtime.device == "cpu":
        return cpu_name()
    return runtime.device


def _version(package: str):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def describe_runtime(runtime: RuntimeConfig, tool=None) -> dict:
    info = {
        "backend": runtime.backend,
        "device": runtime.device,
        "device_name": device_name(runtime),
    }
    if tool is not None:
        info["precision"] = inference_precision(tool)
        if runtime.backend == "onnxruntime":
            info["active_providers"] = tool.session.get_providers()
    return info


def inference_precision(tool) -> str:
    """
    Precisão numérica em que o modelo realmente roda.
    OpenVINO escolhe por dispositivo (GPU e NPU usam FP16 por padrão, CPU usa FP32);
    o onnxruntime roda no tipo do próprio ONNX (FP32 nos modelos da rtmlib).
    """
    compiled = getattr(tool, "compiled_model", None)
    if compiled is not None:
        try:
            base = {"f32": "FP32", "f16": "FP16", "bf16": "BF16"}.get(
                str(compiled.get_property("INFERENCE_PRECISION_HINT").get_type_name()), "?")
        except Exception:
            base = "?"
        # Modelo quantizado (NNCF): as camadas quantizadas rodam em INT8 e o
        # restante na precisão base do dispositivo
        if _is_quantized(tool.onnx_model):
            return f"INT8 (+{base})"
        return base
    return "FP32"


def _is_quantized(model_path: str) -> bool:
    try:
        from openvino import Core
        model = Core().read_model(model_path)
    except Exception:
        return False
    return any(op.get_type_name() == "FakeQuantize" for op in model.get_ops())


def runtime_label(pose_runtime: RuntimeConfig, det_runtime: RuntimeConfig) -> str:
    """'openvino/gpu' ou, no modo híbrido, 'det openvino/npu + pose openvino/gpu'."""
    pose = f"{pose_runtime.backend}/{pose_runtime.device}"
    if det_runtime == pose_runtime:
        return pose
    return f"det {det_runtime.backend}/{det_runtime.device} + pose {pose}"


def describe_system() -> dict:
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_name(),
        "cpu_logical_cores": os.cpu_count(),
        "libraries": {
            pkg: _version(pkg)
            for pkg in ("rtmlib", "onnxruntime", "onnxruntime-gpu", "openvino",
                        "opencv-python", "numpy")
            if _version(pkg)
        },
    }
