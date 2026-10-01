"""Inspeciona os candidatos do rtmlib: checkpoints, input size, nº de keypoints
e quais landmarks do MVP cada um entrega (T3).

Uso: python scripts/inspect_candidates.py input/video.mp4 [indice_do_frame]
"""
import sys
import time

import cv2
import numpy as np
from rtmlib import BodyWithFeet, Wholebody

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "input/video.mp4"
FRAME_INDEX = int(sys.argv[2]) if len(sys.argv) > 2 else 30
DEVICE = "cpu"
BACKEND = "onnxruntime"

# (nome, classe do rtmlib, modo, layout)
CANDIDATES = [
    ("body_with_feet_performance", BodyWithFeet, "performance", "halpe26"),
    ("body_with_feet_balanced", BodyWithFeet, "balanced", "halpe26"),
    ("wholebody_performance", Wholebody, "performance", "coco_wholebody"),
    ("wholebody_balanced", Wholebody, "balanced", "coco_wholebody"),
]

EXPECTED_KPTS = {"halpe26": 26, "coco_wholebody": 133}

# Índice de cada landmark do MVP (lado direito) em cada layout.
# Atenção: a ordem de heel/toe é diferente entre os dois layouts.
# right_toe = dedão (big toe).
MVP_LANDMARKS = {
    "halpe26": {
        "right_shoulder": 6, "right_elbow": 8, "right_wrist": 10,
        "right_hip": 12, "right_knee": 14, "right_ankle": 16,
        "right_heel": 25, "right_toe": 21,
    },
    "coco_wholebody": {
        "right_shoulder": 6, "right_elbow": 8, "right_wrist": 10,
        "right_hip": 12, "right_knee": 14, "right_ankle": 16,
        "right_heel": 22, "right_toe": 20,
    },
}


def read_frame(path, index):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Não foi possível ler o frame {index}")
    return frame


def main():
    frame = read_frame(VIDEO, FRAME_INDEX)
    print(f"Vídeo: {VIDEO} | frame {FRAME_INDEX} | {frame.shape[1]}x{frame.shape[0]}\n")

    for name, cls, mode, layout in CANDIDATES:
        print("=" * 70)
        print(f"Candidato: {name}  ({cls.__name__}, modo={mode})")

        mode_cfg = getattr(cls, "MODE", {}).get(mode, {})
        for key in ("det", "det_input_size", "pose", "pose_input_size"):
            print(f"  {key}: {mode_cfg.get(key, '(não encontrado)')}")

        model = cls(mode=mode, backend=BACKEND, device=DEVICE)
        model(frame)  # warm-up (a 1ª chamada é mais lenta)
        t0 = time.time()
        keypoints, scores = model(frame)
        elapsed = time.time() - t0

        if keypoints is None or len(keypoints) == 0:
            print("  Nenhuma pessoa detectada neste frame. Tente outro índice de frame.\n")
            continue

        n_kpts = keypoints.shape[1]
        flag = "" if n_kpts == EXPECTED_KPTS[layout] else f"  <-- ESPERADO {EXPECTED_KPTS[layout]}"
        print(f"  pessoas detectadas: {len(keypoints)} | keypoints por pessoa: {n_kpts}{flag}")
        print(f"  tempo de inferência (1 frame, CPU): {elapsed:.2f}s")

        # Para inspeção, usa a pessoa com maior confiança média nos landmarks do MVP
        idxs = list(MVP_LANDMARKS[layout].values())
        best = int(np.argmax(scores[:, idxs].mean(axis=1)))

        print(f"  landmarks do MVP (pessoa {best}):")
        for lm, i in MVP_LANDMARKS[layout].items():
            print(f"    {lm:<16} idx={i:<3} conf={scores[best, i]:.2f}")
        print()


if __name__ == "__main__":
    main()