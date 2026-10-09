"""
Entrada do container: processa UM vídeo e grava os resultados em output/<nome_do_video>/.

Uso (dentro do container; normalmente via docker compose, veja docs/DOCKER.md):
    python docker/processar.py meu_video.mp4

O nome é procurado em /app/input. Também aceita um caminho completo.
Não altera nenhum arquivo do projeto: a pasta de saída é aplicada numa cópia temporária da config.
"""

import os
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "docker_cpu.yaml"
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
EXTENSOES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        raise SystemExit("Uso: processar.py <video>   (arquivo dentro de input/)")

    video = Path(sys.argv[1])
    if not video.is_absolute():
        video = INPUT_DIR / video
    if not video.is_file():
        disponiveis = sorted(p.name for p in INPUT_DIR.glob("*") if p.suffix.lower() in EXTENSOES)
        raise SystemExit(
            f"Vídeo não encontrado: {video}\n"
            f"Vídeos em input/: {', '.join(disponiveis) if disponiveis else '(nenhum; coloque o vídeo na pasta input/)'}"
        )

    saida = OUTPUT_DIR / video.stem
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["pose"]["output"]["dir"] = str(saida)
    saida.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="docker_cpu_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"Vídeo: {video}\nSaída: {saida}\nModelos: baixados na 1ª execução (cache em volume)", flush=True)
    os.execv(sys.executable, [sys.executable, str(ROOT / "run_pose.py"), "--video", str(video), "--config", tmp])


if __name__ == "__main__":
    main()