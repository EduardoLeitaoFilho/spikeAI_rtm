"""
Esquema e leitura do arquivo de configuração do pipeline (YAML).

Todo o comportamento do pipeline (modelo, checkpoints, tamanhos de entrada,
limiares de confiança e saídas) vem daqui. O loader valida o arquivo e falha
cedo com uma mensagem clara em caso de chave ausente, desconhecida ou inválida.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import yaml

from keypoint_layouts import DEFAULT_LANDMARKS, KNOWN_LANDMARKS, LAYOUTS, KeypointLayout


# Dispositivos aceitos por cada backend da rtmlib.
# onnxruntime também aceita 'cuda:N' para escolher a GPU pelo índice.
BACKEND_DEVICES = {
    "onnxruntime": ("cpu", "cuda", "rocm", "mps"),
    "openvino": ("cpu", "gpu", "npu"),
    "opencv": ("cpu", "cuda"),
}


class ConfigError(ValueError):
    """Erro de validação do arquivo de configuração."""


# ============================================================
# ESQUEMA
# ============================================================

@dataclass(frozen=True)
class InputSize:
    width: int
    height: int

    def as_tuple(self):
        """Formato (width, height) esperado pela rtmlib."""
        return (self.width, self.height)


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    device: str


@dataclass(frozen=True)
class DetectorConfig:
    model: str
    checkpoint: str
    input_size: InputSize
    runtime: RuntimeConfig  # igual ao da pose, a menos que o YAML defina outro (modo híbrido)


@dataclass(frozen=True)
class ConfidenceThresholds:
    valid: float
    uncertain: float

    def classify(self, conf: float) -> str:
        """
        Status de um landmark a partir da confiança (Threshold Engine):
            conf >= valid              -> 'valid'
            uncertain <= conf < valid  -> 'uncertain'
            conf < uncertain           -> 'missing'
        """
        if conf >= self.valid:
            return "valid"
        if conf >= self.uncertain:
            return "uncertain"
        return "missing"


@dataclass(frozen=True)
class FrameNumberConfig:
    show: bool      # desenha o número do frame no vídeo anotado
    start_at: int   # 0 ou 1: número exibido no primeiro frame


@dataclass(frozen=True)
class OutputConfig:
    dir: Path
    landmarks_csv: Path
    keypoints_json: Path
    events_json: Path
    run_info: Path
    include_skeleton_video: bool
    skeleton_video: Path
    frame_number: FrameNumberConfig


@dataclass(frozen=True)
class PoseConfig:
    model: str
    checkpoint: str
    input_size: InputSize
    detector: DetectorConfig
    runtime: RuntimeConfig
    confidence_thresholds: ConfidenceThresholds
    output: OutputConfig
    keypoint_layout: KeypointLayout
    # Nomes padronizados exportados no keypoints.json, na ordem do YAML
    landmarks: Tuple[str, ...]
    # Observação livre (opcional), gravada no run_info.json e no comparativo
    note: Optional[str] = None


@dataclass(frozen=True)
class InputConfig:
    source: Union[str, int]  # caminho do vídeo ou índice da webcam
    show_preview: bool


@dataclass(frozen=True)
class MetricsConfig:
    # Frames iniciais excluídos das estatísticas de tempo (compilação/alocação
    # de GPU e NPU deixam os primeiros frames bem mais lentos)
    warmup_frames: int


@dataclass(frozen=True)
class PipelineConfig:
    input: InputConfig
    pose: PoseConfig
    metrics: MetricsConfig


# ============================================================
# LEITURA / VALIDAÇÃO
# ============================================================

def _section(data: Any, path: str, keys: set, optional: set = frozenset()) -> dict:
    """Garante que `data` é um dict contendo exatamente `keys` (mais `optional`, se houver)."""
    if not isinstance(data, dict):
        raise ConfigError(f"'{path}' deve ser um mapeamento, recebido: {type(data).__name__}")
    missing = keys - data.keys()
    unknown = data.keys() - keys - optional
    if missing:
        raise ConfigError(f"'{path}': chave(s) obrigatória(s) ausente(s): {sorted(missing)}")
    if unknown:
        raise ConfigError(f"'{path}': chave(s) desconhecida(s): {sorted(unknown)}")
    return data


def _str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{path}' deve ser uma string não vazia")
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{path}' deve ser true/false")
    return value


def _choice(value: Any, path: str, options) -> str:
    value = _str(value, path)
    if value not in options:
        raise ConfigError(f"'{path}' deve ser um de {list(options)}, recebido: '{value}'")
    return value


def _unit_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{path}' deve ser numérico")
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"'{path}' deve estar entre 0 e 1, recebido: {value}")
    return float(value)


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _resolve(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _checkpoint(value: Any, path: str, base_dir: Path) -> str:
    """URL é mantida como está; caminho local é resolvido e precisa existir."""
    value = _str(value, path)
    if _is_url(value):
        return value
    resolved = _resolve(value, base_dir)
    if not resolved.is_file():
        raise ConfigError(f"'{path}': arquivo não encontrado: {resolved}")
    return str(resolved)


def _input_size(data: Any, path: str) -> InputSize:
    data = _section(data, path, {"width", "height"})
    for key in ("width", "height"):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"'{path}.{key}' deve ser um inteiro positivo")
    return InputSize(width=data["width"], height=data["height"])


def _frame_number(data: Any, path: str) -> FrameNumberConfig:
    """Seção opcional; chaves ausentes usam o padrão (mostrar, começando em 1)."""
    data = _section(data, path, set(), optional={"show", "start_at"})
    start_at = data.get("start_at", 1)
    if isinstance(start_at, bool) or start_at not in (0, 1):
        raise ConfigError(f"'{path}.start_at' deve ser 0 ou 1, recebido: {start_at}")
    return FrameNumberConfig(
        show=_bool(data.get("show", True), f"{path}.show"),
        start_at=start_at,
    )


def _landmarks(value: Any, path: str) -> Tuple[str, ...]:
    """
    Lista de nomes padronizados. Cada nome precisa existir em algum layout
    (pega erro de digitação); se o layout do modelo não o tiver, sai como 'missing'.
    """
    if not isinstance(value, list) or not value:
        raise ConfigError(f"'{path}' deve ser uma lista não vazia de nomes")
    names = tuple(_str(v, f"{path}[{i}]") for i, v in enumerate(value))
    unknown = [n for n in names if n not in KNOWN_LANDMARKS]
    if unknown:
        raise ConfigError(f"'{path}': landmark(s) desconhecido(s): {unknown}")
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise ConfigError(f"'{path}': landmark(s) repetido(s): {duplicated}")
    return names


def _parse_input(data: Any, base_dir: Path) -> InputConfig:
    data = _section(data, "input", {"source", "show_preview"})
    source = data["source"]
    if isinstance(source, int) and not isinstance(source, bool):
        parsed_source = source
    else:
        parsed_source = str(_resolve(_str(source, "input.source"), base_dir))
    return InputConfig(
        source=parsed_source,
        show_preview=_bool(data["show_preview"], "input.show_preview"),
    )


def _runtime(data: Any, path: str) -> RuntimeConfig:
    rt = _section(data, path, {"backend", "device"})
    backend = _choice(rt["backend"], f"{path}.backend", BACKEND_DEVICES)
    device = _str(rt["device"], f"{path}.device")
    cuda_index = backend == "onnxruntime" and re.fullmatch(r"cuda:\d+", device)
    if device not in BACKEND_DEVICES[backend] and not cuda_index:
        raise ConfigError(
            f"'{path}.device': '{device}' não é suportado pelo backend '{backend}'. "
            f"Opções: {list(BACKEND_DEVICES[backend])}"
            + (" ou 'cuda:N'" if backend == "onnxruntime" else "")
        )
    return RuntimeConfig(backend=backend, device=device)


def _parse_pose(data: Any, base_dir: Path) -> PoseConfig:
    data = _section(data, "pose", {
        "model", "checkpoint", "input_size", "detector",
        "runtime", "confidence_thresholds", "output",
    }, optional={"keypoint_layout", "landmarks", "note"})

    runtime = _runtime(data["runtime"], "pose.runtime")

    # 'runtime' do detector é opcional: sem ele, o detector usa o mesmo da pose
    det = _section(data["detector"], "pose.detector", {"model", "checkpoint", "input_size"},
                   optional={"runtime"})
    detector = DetectorConfig(
        model=_str(det["model"], "pose.detector.model"),
        checkpoint=_checkpoint(det["checkpoint"], "pose.detector.checkpoint", base_dir),
        input_size=_input_size(det["input_size"], "pose.detector.input_size"),
        runtime=_runtime(det["runtime"], "pose.detector.runtime") if "runtime" in det else runtime,
    )

    th = _section(data["confidence_thresholds"], "pose.confidence_thresholds", {"valid", "uncertain"})
    thresholds = ConfidenceThresholds(
        valid=_unit_float(th["valid"], "pose.confidence_thresholds.valid"),
        uncertain=_unit_float(th["uncertain"], "pose.confidence_thresholds.uncertain"),
    )
    if thresholds.uncertain >= thresholds.valid:
        raise ConfigError(
            "'pose.confidence_thresholds': 'uncertain' deve ser menor que 'valid' "
            f"(uncertain={thresholds.uncertain}, valid={thresholds.valid})"
        )

    out = _section(data["output"], "pose.output", {
        "dir", "landmarks_csv", "events_json", "run_info", "include_skeleton_video", "skeleton_video",
    }, optional={"keypoints_json", "frame_number"})
    out_dir = _resolve(_str(out["dir"], "pose.output.dir"), base_dir)
    output = OutputConfig(
        dir=out_dir,
        landmarks_csv=out_dir / _str(out["landmarks_csv"], "pose.output.landmarks_csv"),
        keypoints_json=out_dir / _str(out.get("keypoints_json", "keypoints.json"),
                                      "pose.output.keypoints_json"),
        events_json=out_dir / _str(out["events_json"], "pose.output.events_json"),
        run_info=out_dir / _str(out["run_info"], "pose.output.run_info"),
        include_skeleton_video=_bool(out["include_skeleton_video"], "pose.output.include_skeleton_video"),
        skeleton_video=out_dir / _str(out["skeleton_video"], "pose.output.skeleton_video"),
        frame_number=_frame_number(out.get("frame_number", {}), "pose.output.frame_number"),
    )

    return PoseConfig(
        model=_str(data["model"], "pose.model"),
        checkpoint=_checkpoint(data["checkpoint"], "pose.checkpoint", base_dir),
        input_size=_input_size(data["input_size"], "pose.input_size"),
        detector=detector,
        runtime=runtime,
        confidence_thresholds=thresholds,
        output=output,
        # Opcional: sem ele, o modelo é tratado como COCO-17 (RTMPose body)
        keypoint_layout=LAYOUTS[_choice(data.get("keypoint_layout", "coco17"),
                                        "pose.keypoint_layout", LAYOUTS)],
        landmarks=_landmarks(data.get("landmarks", list(DEFAULT_LANDMARKS)), "pose.landmarks"),
        note=_str(data["note"], "pose.note") if "note" in data else None,
    )


def _parse_metrics(data: Any) -> MetricsConfig:
    data = _section(data, "metrics", {"warmup_frames"})
    warmup = data["warmup_frames"]
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ConfigError("'metrics.warmup_frames' deve ser um inteiro >= 0")
    return MetricsConfig(warmup_frames=warmup)


def load_config(config_path: Union[str, Path], base_dir: Union[str, Path]) -> PipelineConfig:
    """
    Lê e valida o YAML de configuração.
    Caminhos relativos no arquivo são resolvidos a partir de `base_dir`.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ConfigError(f"Arquivo de configuração não encontrado: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML inválido em {config_path}: {e}") from e

    base_dir = Path(base_dir)
    raw = _section(raw, "<raiz>", {"input", "pose", "metrics"})
    return PipelineConfig(
        input=_parse_input(raw["input"], base_dir),
        pose=_parse_pose(raw["pose"], base_dir),
        metrics=_parse_metrics(raw["metrics"]),
    )
