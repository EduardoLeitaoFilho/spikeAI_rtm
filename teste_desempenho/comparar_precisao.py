"""
Compara os keypoints de cada teste com a referência FP32 (x_cpu: RTMPose-x,
onnxruntime, CPU).

Não há anotação manual do vídeo, então isto mede a CONCORDÂNCIA com o modelo
de referência. Serve para dizer quanto FP16/INT8 (ou o modelo menor) desviam
do RTMPose-x em FP32, não a precisão absoluta em relação à pose real.

Pareamento: a ordem das pessoas pode mudar entre dispositivos (o detector em
FP16 pode detectar em outra ordem ou com pequenas diferenças), então em cada
frame as pessoas são pareadas pela menor distância média entre keypoints.

Keypoints: só os que existem nos dois CSVs (pelo nome da coluna). Entre layouts
diferentes (COCO-17, Halpe26, WholeBody) isso compara o corpo COCO-17 e, quando
os dois têm, os pés.

Métricas (só keypoints com confiança >= 'uncertain' na referência):
    pessoas_pareadas_pct   % das pessoas da referência encontradas no teste
    erro_medio_px          distância média entre keypoints (pixels)
    nme_pct                erro normalizado pelo tamanho da pessoa (%)
    pck_5pct               % de keypoints a menos de 5% do tamanho da pessoa
    delta_confianca        diferença média de confiança (teste - referência)
    eventos                se take_off/landing caíram nos mesmos frames

Uso:
    python teste_desempenho/comparar_precisao.py            # todos os resultados
    python teste_desempenho/comparar_precisao.py x_int8_npu
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "resultados"
REFERENCE = "x_cpu"

# Pares com distância média acima disto (fração do tamanho da pessoa) não são a mesma pessoa
MAX_MATCH_DIST = 0.5
PCK_THRESHOLD = 0.05


def keypoint_names(csv_path: Path):
    """Nomes dos keypoints, na ordem do cabeçalho: frame; person_id; <K x (x, y, conf)>."""
    with open(csv_path, encoding="utf-8", newline="") as f:
        header = next(csv.reader(f, delimiter=";"))
    return [col[:-2] for col in header[2::3]]  # 'left_ankle_x' -> 'left_ankle'


def load_people(csv_path: Path, names):
    """{frame: array (n_pessoas, len(names), 3)} com x, y, conf dos keypoints `names`."""
    all_names = keypoint_names(csv_path)
    cols = [all_names.index(n) for n in names]
    frames = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            kpts = np.asarray(row[2:], dtype=float).reshape(len(all_names), 3)[cols]
            frames.setdefault(int(row[0]), []).append(kpts)
    return {k: np.stack(v) for k, v in frames.items()}


def person_size(kpts):
    """Maior lado da caixa que envolve os keypoints (referência de escala)."""
    xy = kpts[:, :2]
    return max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1.0)


def match_people(ref, cand):
    """Pareamento guloso pela menor distância média entre keypoints."""
    if len(ref) == 0 or len(cand) == 0:
        return []
    dist = np.linalg.norm(ref[:, None, :, :2] - cand[None, :, :, :2], axis=-1).mean(axis=-1)
    pairs, used_r, used_c = [], set(), set()
    for flat in np.argsort(dist, axis=None):
        r, c = np.unravel_index(flat, dist.shape)
        if r in used_r or c in used_c:
            continue
        if dist[r, c] > MAX_MATCH_DIST * person_size(ref[r]):
            break  # ordenado: os próximos são ainda mais distantes
        pairs.append((r, c))
        used_r.add(r)
        used_c.add(c)
    return pairs


def _events(result_dir: Path):
    path = result_dir / "events_ambiguous.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return {e["event"]: e["frame"] for e in json.load(f)}


def compare_events(ref_dir: Path, cand_dir: Path) -> str:
    ref, cand = _events(ref_dir), _events(cand_dir)
    if ref is None or cand is None:
        return ""
    diffs = []
    for name, frame in ref.items():
        if name not in cand:
            diffs.append(f"{name} ausente")
        elif cand[name] != frame:
            diffs.append(f"{name} {cand[name] - frame:+d} frames")
    return "iguais" if not diffs else ", ".join(diffs)


def compare(ref_dir: Path, cand_dir: Path, conf_threshold: float) -> dict:
    ref_csv, cand_csv = ref_dir / "landmarks.csv", cand_dir / "landmarks.csv"
    cand_names = set(keypoint_names(cand_csv))
    names = [n for n in keypoint_names(ref_csv) if n in cand_names]
    ref_frames = load_people(ref_csv, names)
    cand_frames = load_people(cand_csv, names)

    total_ref = matched = 0
    dists, norm_dists, dconf = [], [], []
    for frame, ref in ref_frames.items():
        total_ref += len(ref)
        cand = cand_frames.get(frame, np.empty((0, len(names), 3)))
        for r, c in match_people(ref, cand):
            matched += 1
            visible = ref[r, :, 2] >= conf_threshold
            if not visible.any():
                continue
            d = np.linalg.norm(ref[r, visible, :2] - cand[c, visible, :2], axis=-1)
            dists.append(d)
            norm_dists.append(d / person_size(ref[r]))
            dconf.append(cand[c, visible, 2] - ref[r, visible, 2])

    if not dists:
        return {"pessoas_pareadas_pct": 0.0}
    d, nd, dc = np.concatenate(dists), np.concatenate(norm_dists), np.concatenate(dconf)
    return {
        "pessoas_pareadas_pct": round(matched / total_ref * 100, 1),
        "erro_medio_px": round(float(d.mean()), 2),
        "nme_pct": round(float(nd.mean()) * 100, 2),
        "pck_5pct": round(float((nd <= PCK_THRESHOLD).mean()) * 100, 1),
        "delta_confianca": round(float(dc.mean()), 4),
        "eventos": compare_events(ref_dir, cand_dir),
    }


def precision_columns(name: str, conf_threshold: float = 0.40) -> dict:
    """Colunas de precisão de um teste (vazio se faltar a referência ou o CSV)."""
    ref_dir, cand_dir = RESULTS_DIR / REFERENCE, RESULTS_DIR / name
    if not (ref_dir / "landmarks.csv").exists() or not (cand_dir / "landmarks.csv").exists():
        return {}
    if name == REFERENCE:
        return {"eventos": "referência"}
    return compare(ref_dir, cand_dir, conf_threshold)


def main():
    names = sys.argv[1:] or sorted(p.name for p in RESULTS_DIR.iterdir()
                                   if (p / "landmarks.csv").exists())
    if not (RESULTS_DIR / REFERENCE / "landmarks.csv").exists():
        sys.exit(f"Referência ausente: rode antes  python teste_desempenho/run_benchmark.py {REFERENCE}")
    print(f"Referência: {REFERENCE} (RTMPose-x FP32)\n")
    for name in names:
        print(f"{name:<20} {precision_columns(name)}")


if __name__ == "__main__":
    main()
