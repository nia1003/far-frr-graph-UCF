#!/usr/bin/env python3
"""Cascade vs parallel fusion study (meeting 2026-11-19).

For every test set it
  * enumerates all 6.7 M cascade configurations (5 % grid, per-sample counts)
  * runs the parallel 1 % grid + +-10 % / +-20 % refinement
  * answers three optimisation questions for both architectures
      O1  min FAR + FRR
      O2  min FRR  s.t. FAR <= limit
      O3  min mean latency  s.t. FAR <= limit and FRR <= limit
  * repeats O1/O2 on a stratified 50/50 split (optimise on train, report test)
  * writes FAR-FRR and latency-error frontiers
Results go to fusion-comparison/results/.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from cascade import CascadeTable, enumerate_cascade
from common import DATASETS, DETECTORS, RESULTS_DIR, Data, load, stratified_split, write_csv
from parallel import ParallelGrid, coarse_grid, evaluate_point, refine

FAR_LIMITS = (0.01, 0.05, 0.10, 0.20)
JOINT_LIMITS = (0.05, 0.10, 0.20, 0.30, 0.40)
TEST_SETS = DATASETS + ("balanced_pooled", "balanced_equal_source")
MILP_SUMMARY = RESULTS_DIR.parents[1] / "parallel-fusion/milp/milp-results/milp_summary.csv"


def load_set(name: str) -> Data:
    if name in DATASETS:
        return load(name)
    return load(name, RESULTS_DIR / "test-sets" / f"{name}.csv")


# ---------------------------------------------------------------- selection
def pick(far, frr, latency, objective: str, limit: float | None):
    """Return the flat index of the best point (ties: lower latency, then sum)."""
    far = far.ravel()
    frr = frr.ravel()
    lat = latency.ravel()
    total = far + frr
    if objective == "O1":
        mask = np.ones_like(far, dtype=bool)
        keys = (lat, np.abs(far - frr), total)
    elif objective == "O2":
        mask = far <= limit + 1e-12
        keys = (lat, far, frr)
    elif objective == "O3":
        mask = (far <= limit + 1e-12) & (frr <= limit + 1e-12)
        keys = (total, lat)
    else:
        raise ValueError(objective)
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        return None
    # lexsort: last key is primary
    order = np.lexsort(tuple(k[candidates] for k in keys))
    return int(candidates[order[0]])


def parallel_latency(grid: ParallelGrid, data: Data, mode: str) -> np.ndarray:
    """Mean latency for every grid point; depends only on which weights are > 0."""
    w = np.stack([grid.a, grid.b, 1 - grid.a - grid.b], axis=1) > 1e-9
    codes = w @ np.array([1, 2, 4])
    table = np.zeros(8)
    for code in range(1, 8):
        used = [(code >> i) & 1 == 1 for i in range(3)]
        lat = data.latency_ms[:, used]
        table[code] = (lat.max(axis=1) if mode == "parallel" else lat.sum(axis=1)).mean()
    return table[codes]


def parallel_row(data: Data, grid: ParallelGrid, i: int, stage: str) -> dict:
    w = grid.weights(i)
    point = evaluate_point(data, w, float(grid.gamma[i]))
    return {
        "method": f"parallel ({stage})",
        "config": (f"a_xception={w[0]:.3f}, b_ucf={w[1]:.3f}, sbi={w[2]:.3f}, "
                   f"gamma={grid.gamma[i]:.3f}"),
        "far": point["far"], "frr": point["frr"],
        "far_plus_frr": point["far"] + point["frr"],
        "latency_ms": point["latency_parallel_ms"],
        "latency_sequential_ms": point["latency_sequential_ms"],
        "detectors_used": point["detectors_used"],
        "_w": w, "_gamma": float(grid.gamma[i]),
    }


def cascade_row(table: CascadeTable, i: int) -> dict:
    c = table.config(i)
    return {
        "method": "cascade (5% grid)",
        "config": f"{c['ordering']} | {c['thresholds']}",
        "far": c["far"], "frr": c["frr"], "far_plus_frr": c["far_plus_frr"],
        "latency_ms": c["mean_latency_ms"],
        "latency_sequential_ms": c["mean_latency_ms"],
        "detectors_used": c["ordering"].replace(">", "+"),
        "reach_stage2": c["reach_stage2"], "reach_stage3": c["reach_stage3"],
        "_flat": i,
    }


def best_parallel(data, coarse, objective, limit, latency_mode="parallel"):
    """coarse 1 % -> refine +-10 % -> refine +-20 % (each around the previous best)."""
    rows = []
    lat = parallel_latency(coarse, data, latency_mode)
    i = pick(coarse.far, coarse.frr, lat, objective, limit)
    if i is None:
        return [], {}
    rows.append(parallel_row(data, coarse, i, "1% grid"))
    seconds = {"coarse": coarse.seconds}
    center = (float(coarse.a[i]), float(coarse.b[i]), float(coarse.gamma[i]))
    for radius in (0.10, 0.20):
        fine = refine(data, center, radius=radius, steps=100)
        lat_f = parallel_latency(fine, data, latency_mode)
        j = pick(fine.far, fine.frr, lat_f, objective, limit)
        seconds[f"refine{int(radius*100)}"] = fine.seconds
        row = parallel_row(data, fine, j, f"refine +-{int(radius*100)}%")
        # keep the better of coarse and fine (fine always contains the centre)
        rows.append(row)
    return rows, seconds


def run_objectives(name: str, data: Data, table: CascadeTable, coarse: ParallelGrid,
                   milp: dict | None) -> list[dict]:
    out = []
    specs = [("O1", None)] + [("O2", l) for l in FAR_LIMITS] + [("O3", l) for l in JOINT_LIMITS]
    for objective, limit in specs:
        base = {"test_set": name, "objective": objective,
                "limit": "" if limit is None else limit}
        i = pick(table.far, table.frr, table.mean_latency_ms, objective, limit)
        if i is None:
            out.append({**base, "method": "cascade (5% grid)", "config": "infeasible"})
        else:
            out.append({**base, **cascade_row(table, i)})
        rows, _ = best_parallel(data, coarse, objective, limit)
        if not rows:
            out.append({**base, "method": "parallel", "config": "infeasible"})
        out.extend({**base, **r} for r in rows)
        if objective == "O1" and milp is not None:
            out.append({**base, **milp})
    return out


def milp_row(dataset: str, data: Data) -> dict | None:
    if dataset not in DATASETS or not MILP_SUMMARY.is_file():
        return None
    with MILP_SUMMARY.open(encoding="utf-8-sig", newline="") as f:
        rows = {r["dataset"]: r for r in csv.DictReader(f)}
    if dataset not in rows:
        return None
    r = rows[dataset]
    w = np.array([float(r["alpha_xception"]), float(r["alpha_ucf"]), float(r["alpha_sbi"])])
    point = evaluate_point(data, w, float(r["beta"]))
    status = "proven optimal" if r["optimal"] == "True" else f"gap={float(r['mip_gap']):.2%}"
    return {
        "method": f"parallel (MILP, {status})",
        "config": (f"a_xception={w[0]:.4f}, b_ucf={w[1]:.4f}, sbi={w[2]:.4f}, "
                   f"gamma={float(r['beta']):.4f}"),
        "far": point["far"], "frr": point["frr"],
        "far_plus_frr": point["far"] + point["frr"],
        "latency_ms": point["latency_parallel_ms"],
        "latency_sequential_ms": point["latency_sequential_ms"],
        "detectors_used": point["detectors_used"],
    }


# --------------------------------------------------------------- held-out
def holdout(name: str, data: Data) -> list[dict]:
    train, test = stratified_split(data, seed=0)
    table = enumerate_cascade(train)
    coarse = coarse_grid(train)
    rows = []
    from cascade import ORDERINGS, PAIRS, GRID, evaluate_cascade
    for objective, limit in [("O1", None)] + [("O2", l) for l in FAR_LIMITS]:
        base = {"test_set": name, "objective": objective, "limit": "" if limit is None else limit}
        i = pick(table.far, table.frr, table.mean_latency_ms, objective, limit)
        if i is not None:
            p, a, b, t = np.unravel_index(i, table.fake_accepted.shape)
            o = ORDERINGS[p]
            (l1, h1), (l2, h2) = PAIRS[a], PAIRS[b]
            stages = [(o[0], GRID[l1], GRID[h1]), (o[1], GRID[l2], GRID[h2]),
                      (o[2], GRID[t], GRID[t])]
            ev = evaluate_cascade(test, stages)
            rows.append({**base, "method": "cascade (5% grid)",
                         "config": cascade_row(table, i)["config"],
                         "train_far": float(table.far.flat[i]), "train_frr": float(table.frr.flat[i]),
                         "test_far": ev["far"], "test_frr": ev["frr"],
                         "test_far_plus_frr": ev["far"] + ev["frr"]})
        prow, _ = best_parallel(train, coarse, objective, limit)
        if prow:
            r = prow[-1]
            ev = evaluate_point(test, r["_w"], r["_gamma"])
            rows.append({**base, "method": "parallel (1% + refine)", "config": r["config"],
                         "train_far": r["far"], "train_frr": r["frr"],
                         "test_far": ev["far"], "test_frr": ev["frr"],
                         "test_far_plus_frr": ev["far"] + ev["frr"]})
    return rows


# --------------------------------------------------------------- frontiers
def frontier(far, frr, step=0.005):
    far = far.ravel()
    frr = frr.ravel()
    order = np.argsort(far, kind="stable")
    running = np.minimum.accumulate(frr[order])
    limits = np.round(np.arange(0, 1 + 1e-9, step), 6)
    pos = np.searchsorted(far[order], limits + 1e-12, side="right") - 1
    return limits, np.where(pos >= 0, running[np.maximum(pos, 0)], np.nan)


def latency_frontier(total, lat):
    """Pareto points: lowest FAR+FRR reachable within each mean latency."""
    total = total.ravel()
    lat = lat.ravel()
    order = np.lexsort((total, lat))
    running = np.minimum.accumulate(total[order])
    keep = np.concatenate(([True], running[1:] < running[:-1] - 1e-12))
    return lat[order][keep], running[keep]


def strip_private(rows):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def main() -> None:
    started = time.perf_counter()
    objective_rows, holdout_rows, frontier_rows, latency_rows, timing = [], [], [], [], []
    for name in TEST_SETS:
        data = load_set(name)
        table = enumerate_cascade(data)
        coarse = coarse_grid(data)
        refine_secs = best_parallel(data, coarse, "O1", None)[1]
        timing.append({"test_set": name, "n_videos": len(data.labels),
                       "cascade_configs": table.fake_accepted.size,
                       "cascade_seconds": table.seconds,
                       "parallel_coarse_points": coarse.a.size,
                       "parallel_coarse_seconds": coarse.seconds,
                       "parallel_refine10_seconds": refine_secs.get("refine10"),
                       "parallel_refine20_seconds": refine_secs.get("refine20")})
        objective_rows += run_objectives(name, data, table, coarse, milp_row(name, data))
        holdout_rows += holdout(name, data)

        c_lim, c_frr = frontier(table.far, table.frr)
        p_lim, p_frr = frontier(coarse.far, coarse.frr)
        for l, cf, pf in zip(c_lim, c_frr, p_frr):
            frontier_rows.append({"test_set": name, "far_limit": l,
                                  "cascade_min_frr": cf, "parallel_min_frr": pf})
        lv, lb = latency_frontier(table.far + table.frr, table.mean_latency_ms)
        for l, b in zip(lv, lb):
            latency_rows.append({"test_set": name, "architecture": "cascade",
                                 "latency_ms": l, "min_far_plus_frr": b})
        for mode in ("parallel", "sequential"):
            plat = parallel_latency(coarse, data, mode)
            lv, lb = latency_frontier(coarse.far + coarse.frr, plat)
            for l, b in zip(lv, lb):
                latency_rows.append({"test_set": name, "architecture": f"parallel ({mode} run)",
                                     "latency_ms": l, "min_far_plus_frr": b})
        print(f"{name}: done ({time.perf_counter() - started:.0f}s)", flush=True)

    cols = ["test_set", "objective", "limit", "method", "config", "far", "frr",
            "far_plus_frr", "latency_ms", "latency_sequential_ms", "detectors_used",
            "reach_stage2", "reach_stage3"]
    write_csv(RESULTS_DIR / "objectives.csv", strip_private(objective_rows), cols)
    write_csv(RESULTS_DIR / "holdout.csv", holdout_rows,
              ["test_set", "objective", "limit", "method", "config", "train_far",
               "train_frr", "test_far", "test_frr", "test_far_plus_frr"])
    write_csv(RESULTS_DIR / "far_frr_frontier.csv", frontier_rows)
    write_csv(RESULTS_DIR / "latency_frontier.csv", latency_rows)
    write_csv(RESULTS_DIR / "timing.csv", timing)
    print(f"total {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
