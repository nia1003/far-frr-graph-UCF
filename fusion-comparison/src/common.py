"""Shared loading code and conventions for the cascade vs parallel study.

Conventions (same as the cascade / GP work in this repo):
  * score 0 = Real, 1 = Fake; label 0 = Real, 1 = Fake
  * FAR = fake accepted as real / n_fake   (missed deepfakes)
  * FRR = real rejected as fake / n_real   (false alarms)
Note: parallel-fusion/milp names these the other way round (its "far" is
real rejected / n_real).  FAR + FRR is the same under both conventions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FUSION_DIR = ROOT / "parallel-fusion"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

DETECTORS = ("xception", "ucf", "sbi")
DATASETS = ("FaceForensics++", "Celeb-DF-v2", "DFDC")
LATENCY_KEY = {"FaceForensics++": "FaceForensics", "Celeb-DF-v2": "CelebDF", "DFDC": "DFDC"}


@dataclass
class Data:
    name: str
    sample_ids: list[str]
    source: list[str]
    labels: np.ndarray  # int8, 1 = fake
    scores: np.ndarray  # (N, 3) in DETECTORS order
    latency_ms: np.ndarray  # (N, 3) per-sample latency, DETECTORS order

    @property
    def n_real(self) -> int:
        return int(np.count_nonzero(self.labels == 0))

    @property
    def n_fake(self) -> int:
        return int(np.count_nonzero(self.labels == 1))

    def subset(self, mask: np.ndarray, name: str | None = None) -> "Data":
        index = np.flatnonzero(mask)
        return Data(
            name=name or self.name,
            sample_ids=[self.sample_ids[i] for i in index],
            source=[self.source[i] for i in index],
            labels=self.labels[index],
            scores=self.scores[index],
            latency_ms=self.latency_ms[index],
        )


def read_latency() -> dict[str, dict[str, float]]:
    """Average inference time per detector and dataset (ms), from the repo CSVs."""
    table: dict[str, dict[str, float]] = {}
    for detector in DETECTORS:
        path = ROOT / f"{detector}_average_inference_time.csv"
        with path.open(encoding="utf-8-sig", newline="") as file:
            table[detector] = {
                row["Dataset"]: float(row["Time(ms)"]) for row in csv.DictReader(file)
            }
    return table


_LATENCY_CACHE: dict[str, dict[str, float]] = {}


def latency_vector(dataset: str, table: dict[str, dict[str, float]] | None = None) -> np.ndarray:
    if table is None:
        if not _LATENCY_CACHE:
            _LATENCY_CACHE.update(read_latency())
        table = _LATENCY_CACHE
    key = LATENCY_KEY.get(dataset, "Combined(Macro)")
    return np.array([table[d][key] for d in DETECTORS], dtype=float)


def concat(parts: list[Data], name: str) -> Data:
    return Data(
        name=name,
        sample_ids=[i for p in parts for i in p.sample_ids],
        source=[s for p in parts for s in p.source],
        labels=np.concatenate([p.labels for p in parts]),
        scores=np.vstack([p.scores for p in parts]),
        latency_ms=np.vstack([p.latency_ms for p in parts]),
    )


def load(dataset: str, path: Path | None = None) -> Data:
    path = path or FUSION_DIR / f"{dataset}_fusion.csv"
    ids, source, labels, scores = [], [], [], []
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            ids.append(row["sample_id"])
            source.append(row["dataset"])
            labels.append(int(row["label(ground_truth)"]))
            scores.append([float(row[f"{d}_fake_score"]) for d in DETECTORS])
    return Data(
        name=dataset,
        sample_ids=ids,
        source=source,
        labels=np.asarray(labels, dtype=np.int8),
        scores=np.asarray(scores, dtype=float),
        latency_ms=np.vstack([latency_vector(s) for s in source]),
    )


def stratified_split(data: Data, seed: int = 0, train_fraction: float = 0.5):
    """Stratified (by label and by source dataset) random split."""
    rng = np.random.default_rng(seed)
    train = np.zeros(len(data.labels), dtype=bool)
    keys = np.array([f"{s}|{y}" for s, y in zip(data.source, data.labels)])
    for key in np.unique(keys):
        index = np.flatnonzero(keys == key)
        rng.shuffle(index)
        train[index[: int(round(len(index) * train_fraction))]] = True
    return data.subset(train, data.name + "/train"), data.subset(~train, data.name + "/test")


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in columns})


def _fmt(value):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, np.integer):
        return int(value)
    return value
