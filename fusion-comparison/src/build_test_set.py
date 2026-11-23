#!/usr/bin/env python3
"""Build balanced (≈50% real / 50% fake) test sets from the stored detector scores.

Scores are computed once per video (parallel-fusion/*_fusion.csv); this script
only re-samples rows, so no detector has to be re-run.

Outputs (results/test-sets/):
  balanced_pooled.csv        every source dataset balanced to min(n_real, n_fake)
                             and pooled (DFDC dominates by size)
  balanced_equal_source.csv  the same number of real and fake videos from every
                             source dataset (each source weighs the same)
  composition.csv            per-source counts of all sets
"""

from __future__ import annotations

import csv

import numpy as np

from common import DATASETS, DETECTORS, RESULTS_DIR, Data, concat, load, write_csv

SEED = 0


def balanced(data: Data, per_class: int, rng: np.random.Generator) -> Data:
    mask = np.zeros(len(data.labels), dtype=bool)
    for label in (0, 1):
        index = np.flatnonzero(data.labels == label)
        mask[rng.choice(index, size=per_class, replace=False)] = True
    return data.subset(mask)


def save(data: Data, path) -> None:
    rows = []
    for i in range(len(data.labels)):
        row = {
            "dataset": data.source[i],
            "sample_id": data.sample_ids[i],
            "label(ground_truth)": int(data.labels[i]),
        }
        for j, d in enumerate(DETECTORS):
            row[f"{d}_fake_score"] = format(float(data.scores[i, j]), ".17g")
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    sources = {name: load(name) for name in DATASETS}
    pooled = concat(
        [balanced(d, min(d.n_real, d.n_fake), rng) for d in sources.values()],
        "balanced_pooled",
    )
    smallest = min(min(d.n_real, d.n_fake) for d in sources.values())
    equal = concat(
        [balanced(d, smallest, rng) for d in sources.values()], "balanced_equal_source"
    )
    out = RESULTS_DIR / "test-sets"
    save(pooled, out / "balanced_pooled.csv")
    save(equal, out / "balanced_equal_source.csv")

    rows = []
    for set_name, data in [("original", concat(list(sources.values()), "original")),
                           ("balanced_pooled", pooled),
                           ("balanced_equal_source", equal)]:
        for source in DATASETS:
            mask = np.array([s == source for s in data.source])
            rows.append({
                "set": set_name, "source": source,
                "n_real": int(np.count_nonzero(mask & (data.labels == 0))),
                "n_fake": int(np.count_nonzero(mask & (data.labels == 1))),
            })
        rows.append({"set": set_name, "source": "TOTAL",
                     "n_real": data.n_real, "n_fake": data.n_fake})
    write_csv(out / "composition.csv", rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
