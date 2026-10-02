"""
Comando de execução da inferência de pose (Task 4).

Uso (na raiz do projeto):
    python run_pose.py --video input/video.mp4 --config config/pose_config.yaml
    python run_pose.py --video input/video.mp4 --config config/candidatos/candidato_a.yml

--config  YAML com modelo, checkpoint, limiares e saídas (padrão: config/pose_config.yaml)
--video   vídeo de entrada ou índice da webcam; opcional, substitui input.source do YAML

Trocar de modelo/candidato é só trocar o --config: nada no código muda.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from main import main  # noqa: E402

if __name__ == "__main__":
    main()
