"""
Pipeline de pose com RTMPose (via rtmlib; backend/dispositivo definidos no YAML) + OpenCV.
Mesma estrutura do pipeline com YOLO: lê vídeo/webcam, roda inferência
frame a frame, mostra/salva o vídeo anotado e exporta os keypoints para CSV
(todos os pontos do modelo) e keypoints.json (landmarks padronizados com status).

Requisitos: ver setup.ps1 / requirements.txt (CPU, NPU, GPU Intel)
e requirements-cuda.txt (GPU NVIDIA).

Observação: o RTMPose funciona em duas etapas (detector de pessoa + estimador
de pose). A rtmlib cuida dessas duas etapas internamente e baixa os modelos
ONNX automaticamente na primeira execução (fica em cache local depois).

Todo o comportamento (modelos, checkpoints, tamanhos de entrada, limiares e
saídas) vem do YAML de configuração. Comando de execução (raiz do projeto):
    python run_pose.py --video input/video.mp4 --config config/pose_config.yaml
--video é opcional e substitui input.source do YAML.

Obs: o backend 'opencv' apresenta bugs de compatibilidade (erro no gather layer)
com versões recentes do OpenCV DNN nesses modelos ONNX específicos.
'onnxruntime' é o backend oficialmente mais testado pela rtmlib.
"""

import os
import csv
import time
import json
import argparse
import dataclasses
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rtmlib import RTMPose, YOLOX

from config import ConfidenceThresholds, PipelineConfig, PoseConfig, RuntimeConfig, load_config
from keypoint_layouts import KeypointLayout
from keypoints_export import KeypointsJSONExporter
from runtime_check import (
    DeviceUnavailable, check_runtime, describe_runtime, describe_system, prepare_checkpoint,
    runtime_label, verify_active,
)


# ============================================================
# CONSTANTES DE DESENHO
# ============================================================
# Nomes e esqueleto dos keypoints vêm do layout (src/keypoint_layouts.py),
# escolhido no YAML em pose.keypoint_layout.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pose_config.yaml"

# Cores (BGR) dos keypoints por classe de confiança
KPT_COLORS = {"valid": (0, 255, 0), "uncertain": (0, 200, 255)}
SKELETON_COLOR = (0, 200, 255)
# Pessoas que não são o atleta, quando pose.output.highlight_athlete é true
OTHERS_COLOR = (120, 120, 120)

# Raio dos pontos do corpo e dos pontos de detalhe (rosto/mãos no WholeBody)
KPT_RADIUS = 4
DETAIL_KPT_RADIUS = 2


# ============================================================
# OVERLAY DE NÚMERO DO FRAME
# ============================================================

def draw_frame_number(frame, frame_number: int):
    """
    Desenha o número do frame no canto inferior esquerdo da imagem.
    Usa um fundo preto atrás do texto para garantir legibilidade
    independente da cor do piso/quadra atrás.
    """
    text = str(frame_number)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    margin = 15

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    height = frame.shape[0]
    x = margin
    y = height - margin  # baseline do texto, próximo ao canto inferior esquerdo

    # Retângulo de fundo para contraste
    cv2.rectangle(
        frame,
        (x - 5, y - text_h - 5),
        (x + text_w + 5, y + baseline + 5),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


# ============================================================
# POSE DETECTOR
# ============================================================

class ModelLoadError(RuntimeError):
    """Um modelo não pôde ser carregado/compilado no dispositivo configurado."""


def _load_model(factory, name: str, runtime: RuntimeConfig):
    """Carrega um modelo, identificando qual modelo e dispositivo falharam."""
    try:
        tool = factory()
    except Exception as e:
        cause = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
        raise ModelLoadError(
            f"{name} não pôde ser compilado em {runtime.backend}/{runtime.device}: {cause}"
        ) from e
    verify_active(tool, runtime)
    return tool


class KeypointLayoutError(RuntimeError):
    """O modelo devolveu um número de keypoints diferente do layout configurado."""


def athlete_index(people_keypoints, thresholds: ConfidenceThresholds):
    """
    Índice da pessoa tratada como o atleta no vídeo anotado: a de maior área de
    bounding box, calculada só com os keypoints que não são 'missing'.

    É uma heurística por frame, sem tracking: serve para a inspeção visual da
    Task 7, e não para análise. O keypoints.json continua trazendo todas as
    pessoas, e nada aqui altera os dados exportados.
    """
    best, best_area = None, -1.0
    for i, kpts in enumerate(people_keypoints):
        visible = [(x, y) for x, y, conf in kpts if thresholds.classify(conf) != "missing"]
        if len(visible) < 2:
            continue
        xs = [p[0] for p in visible]
        ys = [p[1] for p in visible]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best, best_area = i, float(area)
    return best


class PoseDetector:
    """
    Carrega o detector de pessoas (YOLOX) e o RTMPose (via rtmlib) e roda a
    inferência em frames individuais. Cada modelo pode rodar em um dispositivo
    diferente (modo híbrido); é o mesmo fluxo de rtmlib.Body, que usa um só.
    """

    def __init__(self, pose_cfg: PoseConfig):
        self.thresholds: ConfidenceThresholds = pose_cfg.confidence_thresholds
        self.layout: KeypointLayout = pose_cfg.keypoint_layout
        self.highlight_athlete = pose_cfg.output.highlight_athlete
        self.pose_name = pose_cfg.model
        det_cfg = pose_cfg.detector
        check_runtime(det_cfg.runtime)
        check_runtime(pose_cfg.runtime)
        det_ckpt = prepare_checkpoint(det_cfg.checkpoint, det_cfg.input_size, det_cfg.runtime)
        pose_ckpt = prepare_checkpoint(pose_cfg.checkpoint, pose_cfg.input_size, pose_cfg.runtime)

        # Inclui a compilação do modelo para GPU/NPU
        start = time.perf_counter()
        self.det_model = _load_model(
            lambda: YOLOX(det_ckpt, model_input_size=det_cfg.input_size.as_tuple(),
                          backend=det_cfg.runtime.backend, device=det_cfg.runtime.device),
            f"{det_cfg.model} (detector)", det_cfg.runtime,
        )
        self.pose_model = _load_model(
            lambda: RTMPose(pose_ckpt, model_input_size=pose_cfg.input_size.as_tuple(),
                            to_openpose=False,
                            backend=pose_cfg.runtime.backend, device=pose_cfg.runtime.device),
            f"{pose_cfg.model} (pose)", pose_cfg.runtime,
        )
        self.load_time_s = time.perf_counter() - start

        # Tempos de inferência por frame (ms), separados por etapa
        self.det_times_ms = []
        self.pose_times_ms = []

    def _infer(self, frame):
        """Detector -> pose, cronometrando cada etapa separadamente."""
        t0 = time.perf_counter()
        bboxes = self.det_model(frame)
        det_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        keypoints, scores = self.pose_model(frame, bboxes=bboxes)
        self.det_times_ms.append(det_ms)
        self.pose_times_ms.append((time.perf_counter() - t0) * 1000)
        self._check_layout(keypoints)
        return keypoints, scores

    def _check_layout(self, keypoints):
        """Falha se o modelo não entrega o número de keypoints do layout do YAML."""
        if keypoints is None or len(keypoints) == 0:
            return
        n_model = keypoints.shape[1]
        if n_model != len(self.layout):
            raise KeypointLayoutError(
                f"{self.pose_name} devolveu {n_model} keypoints, mas pose.keypoint_layout é "
                f"'{self.layout.name}' ({len(self.layout)} keypoints). Ajuste o YAML."
            )

    def process_frame(self, frame):
        """
        Roda a inferência em um frame e retorna:
          - annotated_frame: frame com o esqueleto desenhado
          - people_keypoints: lista de arrays (K, 3) -> (x, y, conf) por pessoa,
            com K = número de keypoints do layout
        """
        keypoints, scores = self._infer(frame)  # keypoints: (num_pessoas, K, 2) | scores: (num_pessoas, K)

        annotated_frame = frame.copy()
        people_keypoints = []

        if keypoints is not None and len(keypoints) > 0:
            for person_xy, person_conf in zip(keypoints, scores):
                person_kpts = np.concatenate(
                    [person_xy, person_conf[:, None]], axis=1
                )  # (K, 3) -> x, y, conf
                people_keypoints.append(person_kpts)

            athlete = athlete_index(people_keypoints, self.thresholds) if self.highlight_athlete else None
            # O atleta é desenhado por último, para ficar por cima das demais pessoas
            for i in sorted(range(len(people_keypoints)), key=lambda i: i == athlete):
                self._draw_person(annotated_frame, people_keypoints[i],
                                  highlight=(athlete is None or i == athlete))

        return annotated_frame, people_keypoints

    def _draw_person(self, frame, person_kpts, highlight=True):
        """
        Desenha os pontos e o esqueleto de uma pessoa no frame.
        Keypoints 'valid' e 'uncertain' são desenhados (com cores diferentes);
        'missing' é omitido, assim como as conexões que dependem dele.
        Com highlight=False a pessoa sai esmaecida (não é o atleta).
        """
        classes = [self.thresholds.classify(conf) for _, _, conf in person_kpts]

        detail_start = self.layout.detail_start
        for idx, ((x, y, _), cls) in enumerate(zip(person_kpts, classes)):
            if cls != "missing":
                is_detail = detail_start is not None and idx >= detail_start
                radius = DETAIL_KPT_RADIUS if is_detail else KPT_RADIUS
                if not highlight:
                    radius = max(1, radius - 2)
                color = KPT_COLORS[cls] if highlight else OTHERS_COLOR
                cv2.circle(frame, (int(x), int(y)), radius, color, -1)

        line_color = SKELETON_COLOR if highlight else OTHERS_COLOR
        thickness = 2 if highlight else 1
        for i, j in self.layout.skeleton:
            if classes[i] != "missing" and classes[j] != "missing":
                xi, yi, _ = person_kpts[i]
                xj, yj, _ = person_kpts[j]
                cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), line_color, thickness)


# ============================================================
# CSV EXPORTER
# ============================================================

class CSVExporter:
    """Acumula os keypoints frame a frame e exporta tudo para CSV ao final."""

    def __init__(self, output_path: str, layout: KeypointLayout):
        self.output_path = output_path
        self.layout = layout
        self.rows = []

    def add_frame(self, frame_index: int, people_keypoints):
        """Adiciona os keypoints de um frame ao buffer (uma linha por pessoa)."""
        for person_id, kpts in enumerate(people_keypoints):
            row = {"frame": frame_index, "person_id": person_id}
            for name, (x, y, conf) in zip(self.layout.names, kpts):
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
        for name in self.layout.names:
            fieldnames += [f"{name}_x", f"{name}_y", f"{name}_conf"]

        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(self.rows)

        print(f"CSV salvo em: {self.output_path} ({len(self.rows)} linhas)")


# ============================================================
# EVENTOS AMBIGUOS (teste a partir do CSV)
# ============================================================

def load_landmarks_from_csv(csv_path: str):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def get_person_tracks(rows):
    tracks = {}
    for row in rows:
        person_id = int(row["person_id"])
        frame = int(row["frame"])
        tracks.setdefault(person_id, []).append((frame, row))
    return tracks


def ankle_midpoint_y(row):
    left_y = float(row.get("left_ankle_y", 0))
    right_y = float(row.get("right_ankle_y", 0))
    return (left_y + right_y) / 2.0


def ankle_confidence(row):
    """Confiança do ponto médio dos tornozelos: a menor entre os dois (elo mais fraco)."""
    left_conf = float(row.get("left_ankle_conf", 0))
    right_conf = float(row.get("right_ankle_conf", 0))
    return min(left_conf, right_conf)


def ankle_series(rows, thresholds: ConfidenceThresholds):
    """
    Série temporal dos tornozelos da pessoa com mais detecções.
    Frames com confiança 'missing' são descartados antes da análise.
    Retorna (frames, heights, confs) alinhados.
    """
    tracks = get_person_tracks(rows)
    if not tracks:
        return [], [], []

    person_id = max(tracks.keys(), key=lambda pid: len(tracks[pid]))
    ordered = sorted(tracks[person_id], key=lambda x: x[0])

    frames, heights, confs = [], [], []
    for frame, row in ordered:
        conf = ankle_confidence(row)
        if thresholds.classify(conf) == "missing":
            continue
        frames.append(frame)
        heights.append(ankle_midpoint_y(row))
        confs.append(conf)

    return frames, heights, confs


def build_event(event_name: str, frame_idx: int, confidence: float,
                thresholds: ConfidenceThresholds, uncertainty: int = 0):
    return {
        "event": event_name,
        "frame": int(frame_idx),
        "uncertainty": int(uncertainty),
        "confidence": round(float(confidence), 4),
        "status": thresholds.classify(confidence),
    }


def detect_takeoff_ambiguous(rows, thresholds: ConfidenceThresholds):
    """Heurística simples para teste do formato: frame de maior elevação dos tornozelos."""
    frames, heights, confs = ankle_series(rows, thresholds)
    if len(heights) < 5:
        return []

    min_idx = min(range(len(heights)), key=lambda i: heights[i])
    start = max(0, min_idx - 2)
    end = min(len(heights), min_idx + 3)
    candidate = min(range(start, end), key=lambda i: heights[i])

    return [build_event("take_off", frames[candidate], confs[candidate], thresholds, 2)]


def detect_landing_ambiguous(rows, thresholds: ConfidenceThresholds):
    """Heurística simples para teste do formato: frame de maior altura antes da queda."""
    frames, heights, confs = ankle_series(rows, thresholds)
    if len(heights) < 5:
        return []

    max_idx = max(range(len(heights)), key=lambda i: heights[i])
    start = max(0, max_idx - 2)
    end = min(len(heights), max_idx + 3)
    candidate = max(range(start, end), key=lambda i: heights[i])

    return [build_event("landing", frames[candidate], confs[candidate], thresholds, 2)]


def generate_ambiguous_events_from_csv(csv_path: str, json_output: str,
                                       thresholds: ConfidenceThresholds):
    rows = load_landmarks_from_csv(csv_path)
    if not rows:
        raise ValueError(f"CSV vazio: {csv_path}")

    events = []
    events.extend(detect_takeoff_ambiguous(rows, thresholds))
    events.extend(detect_landing_ambiguous(rows, thresholds))
    if not events:
        print("Nenhum evento gerado: menos de 5 frames com tornozelos acima do limiar 'uncertain'.")

    os.makedirs(os.path.dirname(json_output), exist_ok=True)
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    print(f"Eventos ambíguos salvos em: {json_output}")
    print(json.dumps(events, ensure_ascii=False, indent=2))
    return events


# ============================================================
# MAIN
# ============================================================
def timing_stats(values_ms, warmup_frames: int) -> dict:
    """
    Estatísticas de tempo (ms) ignorando os `warmup_frames` iniciais.
    Se o vídeo tiver frames de menos, usa todos e sinaliza no resultado.
    """
    if not values_ms:
        return {"frames": 0}
    measured = values_ms[warmup_frames:] or values_ms
    arr = np.asarray(measured)
    return {
        "frames": len(measured),
        "warmup_ignored": len(measured) < len(values_ms),
        "mean": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
        "warmup_mean": round(float(np.mean(values_ms[:warmup_frames])), 2) if warmup_frames and len(values_ms) > warmup_frames else None,
    }


# Código de saída quando o dispositivo/modelo não pode ser usado (não é um bug do pipeline)
EXIT_MODEL_UNAVAILABLE = 3


def _safe_describe(runtime: RuntimeConfig, tool=None) -> dict:
    try:
        return describe_runtime(runtime, tool)
    except Exception:  # dispositivo ausente: não dá para consultar o nome
        return {"backend": runtime.backend, "device": runtime.device, "device_name": None}


def base_run_info(cfg: PipelineConfig, args, started_at, detector=None) -> dict:
    """Campos do run_info.json comuns a execuções completas e falhas de carga."""
    pose_cfg = cfg.pose
    det_cfg = pose_cfg.detector
    det_rt = _safe_describe(det_cfg.runtime, detector.det_model if detector else None)
    pose_rt = _safe_describe(pose_cfg.runtime, detector.pose_model if detector else None)
    if det_cfg.runtime == pose_cfg.runtime:
        device_summary = pose_rt["device_name"]
    else:
        device_summary = f"det: {det_rt['device_name']} + pose: {pose_rt['device_name']}"

    return {
        "started_at": started_at.isoformat(timespec="seconds"),
        "config_file": str(Path(args.config).resolve()),
        "note": pose_cfg.note,
        "system": describe_system(),
        "runtime": {
            "label": runtime_label(pose_cfg.runtime, det_cfg.runtime),
            "device_name": device_summary,
            "hybrid": det_cfg.runtime != pose_cfg.runtime,
            "detector": det_rt,
            "pose": pose_rt,
        },
        "models": {
            "pose": {
                "name": pose_cfg.model,
                "checkpoint": pose_cfg.checkpoint,
                "input_size": pose_cfg.input_size.as_tuple(),
                "keypoint_layout": pose_cfg.keypoint_layout.name,
                "num_keypoints": len(pose_cfg.keypoint_layout),
            },
            "detector": {
                "name": det_cfg.model,
                "checkpoint": det_cfg.checkpoint,
                "input_size": det_cfg.input_size.as_tuple(),
            },
        },
        "confidence_thresholds": {
            "valid": pose_cfg.confidence_thresholds.valid,
            "uncertain": pose_cfg.confidence_thresholds.uncertain,
        },
    }


def write_run_info(out_cfg, run_info: dict):
    os.makedirs(out_cfg.dir, exist_ok=True)
    with open(out_cfg.run_info, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de pose RTMPose")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Caminho do YAML de configuração (padrão: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--video",
        help="Vídeo de entrada (ou índice da webcam, ex.: 0). Substitui input.source do YAML.",
    )
    return parser.parse_args()


def override_video(cfg: PipelineConfig, video: str) -> PipelineConfig:
    """Troca input.source pelo --video (caminho relativo ao diretório atual)."""
    if video.isdigit():
        source = int(video)
    else:
        path = Path(video).resolve()
        if not path.is_file():
            raise SystemExit(f"Vídeo não encontrado: {path}")
        source = str(path)
    return dataclasses.replace(cfg, input=dataclasses.replace(cfg.input, source=source))


def frame_timestamp_ms(cap, frame_index: int, fps: float) -> float:
    """
    Timestamp do frame recém-lido. Em arquivos o OpenCV informa a posição real;
    em webcam ela pode vir zerada, então usa índice / FPS.
    """
    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    if pos_ms > 0 or frame_index == 0:
        return pos_ms
    return frame_index * 1000.0 / fps


def main():
    args = parse_args()
    cfg: PipelineConfig = load_config(args.config, base_dir=PROJECT_ROOT)
    if args.video is not None:
        cfg = override_video(cfg, args.video)
    pose_cfg = cfg.pose
    out_cfg = pose_cfg.output

    started_at = datetime.now().astimezone()
    try:
        detector = PoseDetector(pose_cfg)
    except (DeviceUnavailable, ModelLoadError) as e:
        # Registra a falha no run_info.json para o comparativo mostrar o motivo
        status = "indisponível" if isinstance(e, DeviceUnavailable) else "incompatível"
        run_info = base_run_info(cfg, args, started_at)
        run_info.update({"status": status, "error": str(e)})
        write_run_info(out_cfg, run_info)
        print(f"{status.upper()}: {e}")
        raise SystemExit(EXIT_MODEL_UNAVAILABLE)

    exporter = CSVExporter(str(out_cfg.landmarks_csv), pose_cfg.keypoint_layout)

    cap = cv2.VideoCapture(cfg.input.source)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {cfg.input.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    json_exporter = KeypointsJSONExporter(str(out_cfg.keypoints_json), pose_cfg, metadata={
        "config_file": str(Path(args.config).resolve()),
        "video": str(cfg.input.source),
        "fps": fps,
        "width": width,
        "height": height,
    })

    writer = None
    if out_cfg.include_skeleton_video:
        os.makedirs(out_cfg.dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_cfg.skeleton_video), fourcc, fps, (width, height))

    # Estatísticas de processamento
    frame_index = 0
    frames_com_deteccao = 0
    frames_sem_deteccao = 0
    conf_sum = 0.0
    conf_count = 0
    last_timestamp_ms = 0
    completed = False
    layout_error = None
    start_time = time.perf_counter()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            annotated_frame, people_keypoints = detector.process_frame(frame)

            if out_cfg.frame_number.show:
                draw_frame_number(annotated_frame, frame_index + out_cfg.frame_number.start_at)

            last_timestamp_ms = frame_timestamp_ms(cap, frame_index, fps)
            exporter.add_frame(frame_index, people_keypoints)
            json_exporter.add_frame(frame_index, last_timestamp_ms, people_keypoints)

            if len(people_keypoints) > 0:
                frames_com_deteccao += 1
            else:
                frames_sem_deteccao += 1
            for kpts in people_keypoints:
                conf_sum += float(kpts[:, 2].sum())
                conf_count += len(kpts)

            if writer is not None:
                writer.write(annotated_frame)

            if cfg.input.show_preview:
                cv2.imshow("RTMPose Pipeline", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1

        completed = True

    except KeypointLayoutError as e:
        layout_error = e

    finally:
        elapsed_time = time.perf_counter() - start_time

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        if layout_error is not None:
            # Erro de configuração: nada foi exportado, registra como falha de carga
            run_info = base_run_info(cfg, args, started_at, detector)
            run_info.update({"status": "incompatível", "error": str(layout_error)})
            write_run_info(out_cfg, run_info)
            print(f"INCOMPATÍVEL: {layout_error}")
            raise SystemExit(EXIT_MODEL_UNAVAILABLE)

        exporter.save()
        json_exporter.save()

        csv_path = str(out_cfg.landmarks_csv)
        json_path = str(out_cfg.events_json)
        if os.path.exists(csv_path):
            generate_ambiguous_events_from_csv(csv_path, json_path, pose_cfg.confidence_thresholds)
        else:
            print(f"CSV não encontrado para gerar eventos: {csv_path}")

        # Exibe as informações do vídeo e do processamento
        print("Config: ", args.config)
        print("Video: ", cfg.input.source)
        print("Modelo: ", pose_cfg.model, "+", pose_cfg.detector.model,
              "(", runtime_label(pose_cfg.runtime, pose_cfg.detector.runtime), ")")
        print("Resolução: ", width, "x", height)
        print("FPS: ", fps)
        print("Número de frames processados: ", frame_index)
        print("Frames com pessoa detectada: ", frames_com_deteccao)
        print("Frames sem pessoa detectada: ", frames_sem_deteccao)
        if frame_index > 0:
            print("Taxa de detecção: ", frames_com_deteccao / frame_index * 100, "%")
        print("Duração do vídeo processado: ", last_timestamp_ms / 1000, "segundos")
        print("Tempo real de processamento: ", round(elapsed_time, 2), "segundos")

        warmup = cfg.metrics.warmup_frames
        det_stats = timing_stats(detector.det_times_ms, warmup)
        pose_stats = timing_stats(detector.pose_times_ms, warmup)
        total_ms = [d + p for d, p in zip(detector.det_times_ms, detector.pose_times_ms)]
        total_stats = timing_stats(total_ms, warmup)

        run_info = base_run_info(cfg, args, started_at, detector)
        run_info.update({
            "status": "completed" if completed else "interrupted",
            "video": {
                "source": str(cfg.input.source),
                "width": width,
                "height": height,
                "fps": fps,
                "frames_processed": frame_index,
                "duration_s": round(last_timestamp_ms / 1000, 3),
            },
            "detection": {
                "frames_with_person": frames_com_deteccao,
                "frames_without_person": frames_sem_deteccao,
                "detection_rate_pct": round(frames_com_deteccao / frame_index * 100, 2) if frame_index else None,
                "mean_keypoint_confidence": round(conf_sum / conf_count, 4) if conf_count else None,
            },
            "performance": {
                "model_load_s": round(detector.load_time_s, 3),
                "processing_wall_s": round(elapsed_time, 3),
                "pipeline_fps": round(frame_index / elapsed_time, 2) if elapsed_time > 0 else None,
                "warmup_frames_excluded": warmup,
                "inference_ms": {
                    "detector": det_stats,
                    "pose": pose_stats,
                    "total": total_stats,
                },
                "inference_fps": round(1000 / total_stats["mean"], 2) if total_stats.get("mean") else None,
            },
        })
        write_run_info(out_cfg, run_info)

        perf = run_info["performance"]
        print("Dispositivo: ", run_info["runtime"]["device_name"])
        print("Carga dos modelos: ", perf["model_load_s"], "segundos")
        print("Inferência média por frame: ", total_stats.get("mean"), "ms",
              "(detector", det_stats.get("mean"), "+ pose", pose_stats.get("mean"), ")")
        print("FPS de inferência: ", perf["inference_fps"], "| FPS do pipeline: ", perf["pipeline_fps"])
        print(f"Métricas salvas em: {out_cfg.run_info}")


if __name__ == "__main__":
    main()
