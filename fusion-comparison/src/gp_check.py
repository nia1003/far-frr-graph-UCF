#!/usr/bin/env python3
"""Cascade: compare GP (hyperbola upper bound) and curve-based discrete search
with the empirical per-sample optimum.

Writes results/gp_vs_discrete.csv and results/hyperbola_c.csv.
"""

from __future__ import annotations

import csv

import numpy as np

from cascade import enumerate_cascade, evaluate_cascade
from common import DATASETS, DETECTORS, RESULTS_DIR, ROOT, load, write_csv
from run_study import pick

GP_CONB = {"sbi": 0.324, "xception": 0.171, "ucf": 0.137}  # bmma_gpt_lookup/README_GP_FIX.md
GP_FORENSIC = [("ucf", 0.31857452524688723, 0.4155027098217929),
               ("xception", 0.46465656928991583, 0.5812239401526822),
               ("sbi", 0.3184822912682815, 0.3184822912682815)]
GP_CLAIM = (0.19028726331534682, 0.4794626778672095)  # system FAR, FRR in the lookup table
SHORT = {"FaceForensics++": "faceforensics", "Celeb-DF-v2": "celebdf", "DFDC": "dfdc"}
LIMITS = (0.01, 0.05, 0.10, 0.20)


def detector_curve(scores, labels, d):
    t = np.linspace(0, 1, 1001)
    s = scores[:, d]
    far = np.array([(s[labels == 1] < x).mean() for x in t])   # fakes accepted
    frr = np.array([(s[labels == 0] >= x).mean() for x in t])  # reals rejected
    return t, far, frr


def main() -> None:
    c_rows, rows = [], []
    for name in DATASETS:
        data = load(name)
        for d, det in enumerate(DETECTORS):
            t, far, frr = detector_curve(data.scores, data.labels, d)
            k = int(np.argmin(np.abs(far - frr)))
            prod = far * frr
            c_emp = float(prod.max())
            # How far above the empirical curve the hyperbola FRR = C/FAR lies,
            # averaged over the FAR range where C/FAR <= 1.
            mask = (far > 0) & (c_emp / np.maximum(far, 1e-12) <= 1)
            slack = float(np.mean(c_emp / far[mask] - frr[mask])) if mask.any() else np.nan
            c_rows.append({"dataset": name, "detector": det,
                           "eer": (far[k] + frr[k]) / 2, "eer_squared": ((far[k] + frr[k]) / 2) ** 2,
                           "C_empirical_max_far_x_frr": c_emp,
                           "C_used_by_GP_(all_datasets_max)": GP_CONB[det],
                           "mean_hyperbola_slack_on_frr": slack})

        table = enumerate_cascade(data)
        # 1. GP forensic setting
        stages = [(DETECTORS.index(n), lo, hi) for n, lo, hi in GP_FORENSIC]
        ev = evaluate_cascade(data, stages)
        i = pick(table.far, table.frr, table.mean_latency_ms, "O2", ev["far"])
        rows.append({"dataset": name, "source": "GP lookup (forensic, alpha=0.3)",
                     "far_limit": round(ev["far"], 4),
                     "claimed_far": GP_CLAIM[0], "claimed_frr": GP_CLAIM[1],
                     "empirical_far": ev["far"], "empirical_frr": ev["frr"],
                     "empirical_latency_ms": ev["mean_latency_ms"],
                     "best_empirical_frr_at_same_far": float(table.frr.flat[i]),
                     "best_empirical_config": table.config(i)["ordering"] + " | " + table.config(i)["thresholds"]})
        # 2. curve-based exhaustive search (independence assumption)
        path = ROOT / "exhaustive-search/accuracy-oriented-selection/results/20" / (
            f"accuracy_oriented_selection_{SHORT[name]}.csv")
        with path.open(encoding="utf-8-sig", newline="") as f:
            selected = list(csv.DictReader(f))
        for limit in LIMITS:
            r = min((r for r in selected if abs(float(r["FAR Limit"]) - limit) < 1e-9),
                    key=lambda r: float(r["FRR"]))
            order = sorted((x for x in DETECTORS if int(r[x + "_stage"]) > 0),
                           key=lambda x: int(r[x + "_stage"]))
            st = [(DETECTORS.index(x), float(r[x + "_lower_threshold"]),
                   float(r[x + "_upper_threshold"])) for x in order]
            st[-1] = (st[-1][0], st[-1][1], st[-1][1])
            ev = evaluate_cascade(data, st)
            i = pick(table.far, table.frr, table.mean_latency_ms, "O2", ev["far"])
            j = pick(table.far, table.frr, table.mean_latency_ms, "O2", limit)
            rows.append({"dataset": name, "source": "curve-product exhaustive (5% grid)",
                         "far_limit": limit,
                         "claimed_far": float(r["FAR"]), "claimed_frr": float(r["FRR"]),
                         "empirical_far": ev["far"], "empirical_frr": ev["frr"],
                         "empirical_latency_ms": ev["mean_latency_ms"],
                         "best_empirical_frr_at_same_far": float(table.frr.flat[i]),
                         "best_empirical_frr_at_nominal_limit": float(table.frr.flat[j]),
                         "best_empirical_config": table.config(i)["ordering"] + " | " + table.config(i)["thresholds"]})
    write_csv(RESULTS_DIR / "hyperbola_c.csv", c_rows)
    columns = ["dataset", "source", "far_limit", "claimed_far", "claimed_frr",
               "empirical_far", "empirical_frr", "empirical_latency_ms",
               "best_empirical_frr_at_same_far", "best_empirical_frr_at_nominal_limit",
               "best_empirical_config"]
    write_csv(RESULTS_DIR / "gp_vs_discrete.csv", rows, columns)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "best_empirical_config"})


if __name__ == "__main__":
    main()
