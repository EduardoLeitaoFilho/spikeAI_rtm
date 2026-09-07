"""
Pipeline de pose com RTMPose (via rtmlib, backend OpenCV) + OpenCV.
Mesma estrutura do pipeline com YOLO: lê vídeo/webcam, roda inferência
frame a frame, mostra/salva o vídeo anotado e exporta os keypoints para CSV.

Requisitos:
    pip install rtmlib opencv-python opencv-contrib-python numpy onnxruntime

Observação: o RTMPose funciona em duas etapas (detector de pessoa + estimador
de pose). A rtmlib cuida dessas duas etapas internamente e baixa os modelos
ONNX automaticamente na primeira execução (fica em cache local depois).
"""

import os
import csv
import time
import cv2
import numpy as np
from rtmlib import Body


# ============================================================
# CONFIG
# ============================================================

# Fonte do vídeo: 0 para webcam, ou caminho de um arquivo .mp4
VIDEO_SOURCE = "../input/video.mp4"

# Diretório e nomes de saída
OUTPUT_DIR = "../output"
OUTPUT_CSV_NAME = "landmarks.csv"
OUTPUT_VIDEO_NAME = "annotated.mp4"

# Se True, salva o vídeo anotado junto com o CSV
SAVE_VIDEO = True

# Se True, mostra a janela do OpenCV durante o processamento
SHOW_PREVIEW = False

# 'performance' (mais preciso, mais lento), 'balanced' ou 'lightweight' (mais rápido)
POSE_MODE = "performance"

# 'cpu', 'cuda' ou 'mps'
DEVICE = "cpu"

# 'opencv', 'onnxruntime' ou 'openvino'
# Obs: o backend 'opencv' apresenta bugs de compatibilidade (erro no gather layer)
# com versões recentes do OpenCV DNN nesses modelos ONNX específicos.
# 'onnxruntime' é o backend oficialmente mais testado pela rtmlib.
BACKEND = "onnxruntime"

# Confiança mínima para considerar um keypoint válido (usado na visualização)
KPT_CONF_THRESHOLD = 0.5

# Nomes dos 17 keypoints no formato COCO, na ordem retornada pelo RTMPose (Body)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Conexões do esqueleto COCO (pares de índices) para desenhar manualmente
SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),        # braços e ombros
    (5, 11), (6, 12), (11, 12),                      # tronco
    (11, 13), (13, 15), (12, 14), (14, 16),          # pernas
    (0, 1), (0, 2), (1, 3), (2, 4),                  # rosto
]


# ============================================================
# POSE DETECTOR
# ============================================================

class PoseDetector:
    """Carrega o RTMPose (via rtmlib) e roda inferência em frames individuais."""

    def __init__(self, mode: str, backend: str, device: str):
        self.model = Body(
            mode=mode,
            backend=backend,
            device=device,
            to_openpose=False,
        )

    def process_frame(self, frame):
        """
        Roda a inferência em um frame e retorna:
          - annotated_frame: frame com o esqueleto desenhado
          - people_keypoints: lista de arrays (17, 3) -> (x, y, conf) por pessoa
        """
        keypoints, scores = self.model(frame)  # keypoints: (num_pessoas, 17, 2) | scores: (num_pessoas, 17)

        annotated_frame = frame.copy()
        people_keypoints = []

        if keypoints is not None and len(keypoints) > 0:
            for person_xy, person_conf in zip(keypoints, scores):
                person_kpts = np.concatenate(
                    [person_xy, person_conf[:, None]], axis=1
                )  # (17, 3) -> x, y, conf
                people_keypoints.append(person_kpts)
                self._draw_person(annotated_frame, person_kpts)

        return annotated_frame, people_keypoints

    def _draw_person(self, frame, person_kpts):
        """Desenha os pontos e o esqueleto de uma pessoa no frame."""
        for x, y, conf in person_kpts:
            if conf >= KPT_CONF_THRESHOLD:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        for i, j in SKELETON_CONNECTIONS:
            xi, yi, ci = person_kpts[i]
            xj, yj, cj = person_kpts[j]
            if ci >= KPT_CONF_THRESHOLD and cj >= KPT_CONF_THRESHOLD:
                cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), (0, 200, 255), 2)


# ============================================================
# CSV EXPORTER
# ============================================================

class CSVExporter:
    """Acumula os keypoints frame a frame e exporta tudo para CSV ao final."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.rows = []

    def add_frame(self, frame_index: int, people_keypoints):
        """Adiciona os keypoints de um frame ao buffer (uma linha por pessoa)."""
        for person_id, kpts in enumerate(people_keypoints):
            row = {"frame": frame_index, "person_id": person_id}
            for name, (x, y, conf) in zip(KEYPOINT_NAMES, kpts):
                row[f"{name}_x"] = float(x)
                row[f"{name}_y"] = float(y)
                row[f"{name}_conf"] = float(conf)
            self.rows.append(row)

    def save(self):
        if not self.rows:
            print("Nenhum keypoint coletado, CSV não foi gerado.")
            return

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        fieldnames = ["frame", "person_id"]
        for name in KEYPOINT_NAMES:
            fieldnames += [f"{name}_x", f"{name}_y", f"{name}_conf"]

        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(self.rows)

        print(f"CSV salvo em: {self.output_path} ({len(self.rows)} linhas)")


# ============================================================
# MAIN
# ============================================================

def main():
    detector = PoseDetector(mode=POSE_MODE, backend=BACKEND, device=DEVICE)
    exporter = CSVExporter(os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME))

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {VIDEO_SOURCE}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if SAVE_VIDEO:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        video_path = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_NAME)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Estatísticas de processamento
    frame_index = 0
    frames_com_deteccao = 0
    frames_sem_deteccao = 0
    last_timestamp_ms = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            annotated_frame, people_keypoints = detector.process_frame(frame)

            exporter.add_frame(frame_index, people_keypoints)

            if len(people_keypoints) > 0:
                frames_com_deteccao += 1
            else:
                frames_sem_deteccao += 1

            last_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            if writer is not None:
                writer.write(annotated_frame)

            if SHOW_PREVIEW:
                cv2.imshow("RTMPose Pipeline", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1

    finally:
        elapsed_time = time.time() - start_time

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        exporter.save()

        # Exibe as informações do vídeo e do processamento
        print("Video: ", VIDEO_SOURCE)
        print("Modelo: RTMPose (", POSE_MODE, "/", BACKEND, ")")
        print("Resolução: ", width, "x", height)
        print("FPS: ", fps)
        print("Número de frames processados: ", frame_index)
        print("Frames com pessoa detectada: ", frames_com_deteccao)
        print("Frames sem pessoa detectada: ", frames_sem_deteccao)
        if frame_index > 0:
            print("Taxa de detecção: ", frames_com_deteccao / frame_index * 100, "%")
        print("Duração do vídeo processado: ", last_timestamp_ms / 1000, "segundos")
        print("Tempo real de processamento: ", round(elapsed_time, 2), "segundos")


if __name__ == "__main__":
    main()