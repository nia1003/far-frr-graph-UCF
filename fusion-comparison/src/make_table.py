#!/usr/bin/env python3
"""Rebuild the method-comparison table (Dataset x Method: FAR, FRR, FAR+FRR, latency).

All rates use ONE convention:
  FAR = fake accepted as real / n_fake,  FRR = real rejected as fake / n_real.
parallel-fusion/milp/*.csv store the opposite naming, so their far/frr are
swapped here.  The old table mixed both conventions (the FF++ MILP row was
not swapped while the Celeb-DF / DFDC rows were).

Latency of a parallel method: every detector with weight > 0 runs on every
video.  "parallel" = the slowest of them (detectors run concurrently),
"sequential" = their sum (one GPU, one after the other).  The old table used
the mean of the three detector latencies, which corresponds to neither.
"""

from __future__ import annotations

import csv

import numpy as np

from common import DATASETS, DETECTORS, FUSION_DIR, RESULTS_DIR, load, write_csv
from parallel import evaluate_point

ORDER = ["Celeb-DF-v2", "DFDC", "FaceForensics++"]


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def weights(r, keys=("alpha_xception", "alpha_ucf", "alpha_sbi")):
    return np.array([float(r[k]) for k in keys])


def main() -> None:
    milp = {r["dataset"]: r for r in read(FUSION_DIR / "milp/milp-results/milp_summary.csv")}
    fixed = {r["dataset"]: r for r in read(FUSION_DIR / "milp/fixed-ffpp-results/fixed_parameters_summary.csv")}
    base = read(FUSION_DIR / "baseline/results/baseline_summary.csv")
    obj = read(RESULTS_DIR / "objectives.csv")
    rows = []
    for name in ORDER:
        data = load(name)

        def add(method, far, frr, lat_par, lat_seq, note=""):
            rows.append({"dataset": name, "method": method, "FAR": far, "FRR": frr,
                         "FAR+FRR": far + frr, "latency_parallel_ms": lat_par,
                         "latency_sequential_ms": lat_seq, "note": note})

        r = milp[name]
        p = evaluate_point(data, weights(r), float(r["beta"]))
        status = "proven optimal" if r["optimal"] == "True" else f"time limit, gap {float(r['mip_gap']):.1%}"
        add("MILP (optimised on this dataset)", p["far"], p["frr"],
            p["latency_parallel_ms"], p["latency_sequential_ms"], status)
        if name != "FaceForensics++":
            f = fixed[name]
            p = evaluate_point(data, weights(f), float(f["beta"]))
            add("MILP (FF++ parameters, cross-dataset)", p["far"], p["frr"],
                p["latency_parallel_ms"], p["latency_sequential_ms"],
                "weights/threshold learned on FF++")
        for b in (x for x in base if x["dataset"] == name):
            if b["method"] == "score_average":
                p = evaluate_point(data, np.full(3, 1 / 3), float(b["score_threshold"]))
                add("Average Score", p["far"], p["frr"], p["latency_parallel_ms"],
                    p["latency_sequential_ms"], "equal weights, threshold 0.5")
            else:
                votes = (data.scores >= 0.5).sum(axis=1) >= int(b["minimum_fake_votes"])
                far = float(np.mean(~votes[data.labels == 1]))
                frr = float(np.mean(votes[data.labels == 0]))
                lat = data.latency_ms
                add("Relative Majority", far, frr, float(lat.max(axis=1).mean()),
                    float(lat.sum(axis=1).mean()), "2 of 3 detectors say fake")
        c = next(o for o in obj if o["test_set"] == name and o["objective"] == "O1"
                 and o["method"].startswith("cascade"))
        add("Cascade (5% grid, per-video search)", float(c["far"]), float(c["frr"]),
            float(c["latency_ms"]), float(c["latency_ms"]), c["config"])
        old = float(np.mean(data.latency_ms[0]))
        rows.append({"dataset": name, "method": "(old table latency = mean of 3 detectors)",
                     "latency_parallel_ms": old, "latency_sequential_ms": old})
    write_csv(RESULTS_DIR / "method_table.csv", rows)

    lines = ["| Dataset | Method | FAR | FRR | FAR + FRR | Latency, parallel run (ms) | Latency, sequential run (ms) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["method"].startswith("(old"):
            continue
        lines.append(f"| {r['dataset']} | {r['method']} | {r['FAR']:.4f} | {r['FRR']:.4f} | "
                     f"{r['FAR+FRR']:.4f} | {r['latency_parallel_ms']:.1f} | {r['latency_sequential_ms']:.1f} |")
    (RESULTS_DIR / "method_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
