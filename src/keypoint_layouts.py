"""
Layouts de keypoints suportados pelo pipeline (nomes, ordem e esqueleto).

O layout é escolhido no YAML (pose.keypoint_layout) e precisa corresponder ao
modelo do checkpoint; o pipeline confere o número de keypoints na 1ª inferência.

Os 17 primeiros keypoints são os do COCO, na mesma ordem, em todos os layouts.

Também define os nomes padronizados do keypoints.json (pose.landmarks): são os
nomes dos layouts, mais os aliases de STANDARD_ALIASES (ex.: right_toe ->
right_big_toe), para que o mesmo nome funcione com qualquer checkpoint.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class KeypointLayout:
    name: str
    names: Tuple[str, ...]
    skeleton: Tuple[Tuple[int, int], ...]  # pares de índices ligados no desenho
    # A partir deste índice (rosto/mãos) os pontos são desenhados menores
    detail_start: Optional[int] = None

    def __len__(self):
        return len(self.names)

    def index_of(self, standard_name: str) -> Optional[int]:
        """Índice do landmark padronizado neste layout, ou None se o modelo não o fornece."""
        name = STANDARD_ALIASES.get(standard_name, standard_name)
        return self.names.index(name) if name in self.names else None


COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

COCO17_SKELETON = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),        # braços e ombros
    (5, 11), (6, 12), (11, 12),                      # tronco
    (11, 13), (13, 15), (12, 14), (14, 16),          # pernas
    (0, 1), (0, 2), (1, 3), (2, 4),                  # rosto
)

# Halpe26 (rtmlib BodyWithFeet): COCO-17 + cabeça, pescoço, quadril e pés.
HALPE26_NAMES = COCO17_NAMES + (
    "head", "neck", "hip",                                      # 17-19
    "left_big_toe", "right_big_toe",                            # 20-21
    "left_small_toe", "right_small_toe",                        # 22-23
    "left_heel", "right_heel",                                  # 24-25
)

HALPE26_SKELETON = COCO17_SKELETON + (
    (17, 18), (18, 19),                              # cabeça - pescoço - quadril
    (15, 20), (15, 22), (15, 24),                    # pé esquerdo
    (16, 21), (16, 23), (16, 25),                    # pé direito
)


def _hand(prefix: str):
    """21 pontos da mão: punho, depois polegar, indicador, médio, anelar e mínimo (4 cada)."""
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    return (f"{prefix}_wrist",) + tuple(f"{prefix}_{f}_{j}" for f in fingers for j in range(1, 5))


def _hand_skeleton(offset: int):
    pairs = []
    for finger in range(5):
        base = 1 + finger * 4
        pairs.append((0, base))
        pairs += [(base + j, base + j + 1) for j in range(3)]
    return tuple((offset + i, offset + j) for i, j in pairs)


# COCO-WholeBody (rtmlib Wholebody / RTMW): COCO-17 + pés + 68 do rosto + 2 x 21 das mãos.
# Atenção: a ordem dos pés é diferente da Halpe26.
WHOLEBODY_FACE_START = 23
WHOLEBODY_LEFT_HAND_START = 91
WHOLEBODY_RIGHT_HAND_START = 112

COCO_WHOLEBODY_NAMES = (
    COCO17_NAMES
    + ("left_big_toe", "left_small_toe", "left_heel",           # 17-19
       "right_big_toe", "right_small_toe", "right_heel")        # 20-22
    + tuple(f"face_{i}" for i in range(68))                     # 23-90
    + _hand("left_hand")                                        # 91-111
    + _hand("right_hand")                                       # 112-132
)

COCO_WHOLEBODY_SKELETON = (
    COCO17_SKELETON
    + ((15, 17), (15, 18), (15, 19),                 # pé esquerdo
       (16, 20), (16, 21), (16, 22))                 # pé direito
    + _hand_skeleton(WHOLEBODY_LEFT_HAND_START)
    + _hand_skeleton(WHOLEBODY_RIGHT_HAND_START)
)


LAYOUTS = {
    "coco17": KeypointLayout("coco17", COCO17_NAMES, COCO17_SKELETON),
    "halpe26": KeypointLayout("halpe26", HALPE26_NAMES, HALPE26_SKELETON),
    "coco_wholebody": KeypointLayout(
        "coco_wholebody", COCO_WHOLEBODY_NAMES, COCO_WHOLEBODY_SKELETON,
        detail_start=WHOLEBODY_FACE_START,
    ),
}

# Nomes padronizados (keypoints.json) que diferem do nome no layout.
# 'toe' é o dedão (big toe) nos layouts que distinguem os dedos.
STANDARD_ALIASES = {
    "left_toe": "left_big_toe",
    "right_toe": "right_big_toe",
}

# Todo nome padronizado aceito no YAML: existe em algum layout (direto ou por alias)
KNOWN_LANDMARKS = (
    set(STANDARD_ALIASES)
    | {n for layout in LAYOUTS.values() for n in layout.names}
)

# Landmarks do MVP (lado direito): padrão de pose.landmarks
DEFAULT_LANDMARKS = (
    "right_shoulder", "right_elbow", "right_wrist", "right_hip",
    "right_knee", "right_ankle", "right_heel", "right_toe",
)

assert len(LAYOUTS["coco17"]) == 17
assert len(LAYOUTS["halpe26"]) == 26
assert len(LAYOUTS["coco_wholebody"]) == 133
