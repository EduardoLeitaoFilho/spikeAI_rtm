"""
Exportação do keypoints.json (Task 6).

Para cada frame: índice, timestamp e, para cada pessoa detectada, os landmarks
padronizados de pose.landmarks com x, y, confidence e status (Task 5).

Regras:
  - status vem de ConfidenceThresholds.classify (valid / uncertain / missing);
  - status 'missing' -> x e y = null (nunca (0, 0));
  - landmark que o layout do modelo não fornece (ex.: right_heel em coco17)
    -> x, y e confidence = null, status 'missing'.
"""

import json
import os
from typing import Optional

from config import ConfidenceThresholds, PoseConfig


class KeypointsJSONExporter:
    """Acumula os frames e grava o keypoints.json ao final."""

    def __init__(self, output_path: str, pose_cfg: PoseConfig, metadata: Optional[dict] = None):
        self.output_path = output_path
        self.thresholds: ConfidenceThresholds = pose_cfg.confidence_thresholds
        layout = pose_cfg.keypoint_layout
        # Nome padronizado -> índice no layout do modelo (None se o modelo não fornece)
        self.landmark_index = {name: layout.index_of(name) for name in pose_cfg.landmarks}
        self.metadata = {
            "model": pose_cfg.model,
            "keypoint_layout": layout.name,
            "landmarks": list(pose_cfg.landmarks),
            "not_provided_by_model": [n for n, i in self.landmark_index.items() if i is None],
            "confidence_thresholds": {
                "valid": self.thresholds.valid,
                "uncertain": self.thresholds.uncertain,
            },
            **(metadata or {}),
        }
        self.frames = []

    def _landmark(self, kpts, index: Optional[int]) -> dict:
        if index is None:
            return {"x": None, "y": None, "confidence": None, "status": "missing"}
        x, y, conf = (float(v) for v in kpts[index])
        status = self.thresholds.classify(conf)
        if status == "missing":
            x = y = None
        return {
            "x": None if x is None else round(x, 2),
            "y": None if y is None else round(y, 2),
            "confidence": round(conf, 4),
            "status": status,
        }

    def add_frame(self, frame_index: int, timestamp_ms: float, people_keypoints):
        """people_keypoints: lista de arrays (K, 3) -> (x, y, conf), um por pessoa."""
        self.frames.append({
            "frame_index": int(frame_index),
            "timestamp_ms": round(float(timestamp_ms), 2),
            "people": [
                {
                    # Ordem de detecção no frame; não identifica a mesma pessoa entre frames
                    "person_index": person_index,
                    "keypoints": {
                        name: self._landmark(kpts, index)
                        for name, index in self.landmark_index.items()
                    },
                }
                for person_index, kpts in enumerate(people_keypoints)
            ],
        })

    def save(self):
        if not self.frames:
            print("Nenhum frame processado, keypoints.json não foi gerado.")
            return
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": self.metadata, "frames": self.frames},
                      f, ensure_ascii=False, indent=2)
        print(f"keypoints.json salvo em: {self.output_path} ({len(self.frames)} frames)")
