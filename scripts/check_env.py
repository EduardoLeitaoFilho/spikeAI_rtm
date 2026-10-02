"""Imprime as versoes do ambiente (para registrar na T1) e avisa de problemas."""
import platform
import sys
from importlib import metadata

PKGS = [
    "rtmlib", "onnxruntime", "onnxruntime-gpu",
    "opencv-python", "opencv-contrib-python", "opencv-python-headless",
    "numpy", "pyyaml", "openvino",
]
REQUIRED = ["rtmlib", "opencv-python", "opencv-contrib-python", "numpy", "pyyaml"]


def version_of(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def main():
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"SO: {platform.platform()}")

    print("\nPacotes instalados:")
    installed = {}
    for pkg in PKGS:
        v = version_of(pkg)
        if v:
            installed[pkg] = v
            print(f"  {pkg}=={v}")

    problems = []
    for pkg in REQUIRED:
        if pkg not in installed:
            problems.append(f"falta instalar: {pkg}")
    if "onnxruntime" not in installed and "onnxruntime-gpu" not in installed:
        problems.append("falta instalar: onnxruntime (ou onnxruntime-gpu)")

    opencv_versions = {p: v for p, v in installed.items() if p.startswith("opencv")}
    if len(set(opencv_versions.values())) > 1:
        problems.append(f"versoes de OpenCV divergentes: {opencv_versions}")
    if "onnxruntime" in installed and "onnxruntime-gpu" in installed:
        problems.append("onnxruntime e onnxruntime-gpu instalados juntos")

    try:
        import onnxruntime as ort
        print(f"\nONNX Runtime providers: {ort.get_available_providers()}")
    except ImportError:
        print("\nONNX Runtime nao importavel")

    # OpenVINO e opcional (ausente no ambiente CUDA); necessario para GPU/NPU Intel
    if "openvino" in installed:
        try:
            import openvino as ov
            print(f"OpenVINO devices: {ov.Core().available_devices}")
        except ImportError:
            print("OpenVINO nao importavel")

    try:
        from rtmlib import Body  # noqa: F401
        print("rtmlib importado com sucesso")
    except ImportError as e:
        print(f"rtmlib nao importavel: {e}")

    if problems:
        print("\nPROBLEMAS ENCONTRADOS:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nAmbiente OK")


if __name__ == "__main__":
    main()