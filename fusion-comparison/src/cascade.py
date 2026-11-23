"""Empirical (per-sample) exhaustive search of the 3-stage dual-threshold cascade.

Decision rule (same as audit_fusion_comparison.py and the GP lookup table):
  stage i < last: score <  L_i -> Real (stop)
                  score >= H_i -> Fake (stop)
                  otherwise    -> next stage
  last stage:     score <  T   -> Real, else Fake
Thresholds take the 21 values 0.00, 0.05, ..., 1.00 with L_i <= H_i, so there
are 231 * 231 * 21 = 1,120,581 legal combinations per ordering and 6 orderings
(6.7 M in total).  Choosing L_i == H_i makes stage i final, so all 15 orderings
of length 1-3 are contained in the 6 full orderings.

Unlike exhaustive-search/ (which multiplies per-detector sweep curves and
therefore assumes the detectors are independent), every FAR / FRR / latency
here is counted on the actual videos, so correlations between detectors are
included.  Each video's three scores are read once; changing thresholds only
re-counts.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

from common import DETECTORS, Data

GRID = np.round(np.arange(21) / 20.0, 10)  # 0.00 .. 1.00, step 0.05
PAIRS = np.array([(l, h) for l in range(21) for h in range(l, 21)], dtype=np.int16)
ORDERINGS = list(itertools.permutations(range(3)))


@dataclass
class CascadeTable:
    """All configurations of one dataset.  Arrays have shape (6, 231, 231, 21)."""

    data_name: str
    n_real: int
    n_fake: int
    fake_accepted: np.ndarray  # number of fakes judged Real (FAR numerator)
    real_rejected: np.ndarray  # number of reals judged Fake (FRR numerator)
    mean_latency_ms: np.ndarray  # average sequential latency per video
    reach2: np.ndarray  # videos that needed stage 2
    reach3: np.ndarray  # videos that needed stage 3
    seconds: float

    @property
    def far(self) -> np.ndarray:
        return self.fake_accepted / self.n_fake

    @property
    def frr(self) -> np.ndarray:
        return self.real_rejected / self.n_real

    def config(self, flat_index: int) -> dict:
        p, a, b, t = np.unravel_index(flat_index, self.fake_accepted.shape)
        order = ORDERINGS[p]
        l1, h1 = PAIRS[a]
        l2, h2 = PAIRS[b]
        names = [DETECTORS[i] for i in order]
        # Report only the stages that are actually reachable.
        stages = [(names[0], GRID[l1], GRID[h1])]
        if l1 != h1:
            stages.append((names[1], GRID[l2], GRID[h2]))
            if l2 != h2:
                stages.append((names[2], GRID[t], GRID[t]))
        # A stage whose L == H decides every video, so it is the last stage.
        return {
            "ordering": ">".join(s[0] for s in stages),
            "stages": len(stages),
            "thresholds": "; ".join(
                f"{n}:L={lo:.2f},H={hi:.2f}" if i < len(stages) - 1 else f"{n}:T={lo:.2f}"
                for i, (n, lo, hi) in enumerate(stages)
            ),
            "far": float(self.far.flat[flat_index]),
            "frr": float(self.frr.flat[flat_index]),
            "far_plus_frr": float(self.far.flat[flat_index] + self.frr.flat[flat_index]),
            "mean_latency_ms": float(self.mean_latency_ms.flat[flat_index]),
            "reach_stage2": int(self.reach2.flat[flat_index]),
            "reach_stage3": int(self.reach3.flat[flat_index]),
        }

    def evaluate_flat(self, flat_index: int) -> tuple[int, int]:
        return int(self.fake_accepted.flat[flat_index]), int(self.real_rejected.flat[flat_index])


def enumerate_cascade(data: Data) -> CascadeTable:
    started = time.perf_counter()
    bins = np.searchsorted(GRID, data.scores, side="right")  # 0..21; s < GRID[k] <=> bin <= k
    labels = data.labels.astype(np.int64)
    n = len(labels)
    shape = (len(ORDERINGS), len(PAIRS), len(PAIRS), 21)
    fake_acc = np.empty(shape, dtype=np.int32)
    real_rej = np.empty(shape, dtype=np.int32)
    latency_sum = np.empty(shape, dtype=np.float64)
    reach2 = np.empty(shape, dtype=np.int32)
    reach3 = np.empty(shape, dtype=np.int32)

    l2 = PAIRS[:, 0].astype(np.int64)
    h2 = PAIRS[:, 1].astype(np.int64)
    for p, order in enumerate(ORDERINGS):
        b1 = bins[:, order[0]]
        b2 = bins[:, order[1]]
        b3 = bins[:, order[2]]
        lat = data.latency_ms[:, list(order)]
        base_latency = float(lat[:, 0].sum())
        for a, (l1, h1) in enumerate(PAIRS):
            real1 = b1 <= l1
            fake1 = b1 >= h1 + 1
            pass1 = ~(real1 | fake1)
            fa1 = int(np.count_nonzero(real1 & (labels == 1)))
            fr1 = int(np.count_nonzero(fake1 & (labels == 0)))
            # Histogram of stage-2/3 bins of the videos that reach stage 2.
            idx = np.flatnonzero(pass1)
            hist = np.zeros((2, 22, 22), dtype=np.int64)
            lat3 = np.zeros((22, 22), dtype=np.float64)
            lat2_by_b2 = np.zeros(22, dtype=np.float64)
            if idx.size:
                np.add.at(hist, (labels[idx], b2[idx], b3[idx]), 1)
                np.add.at(lat3, (b2[idx], b3[idx]), lat[idx, 2])
                np.add.at(lat2_by_b2, b2[idx], lat[idx, 1])
            # cum[y, k, b3] = videos with b2 <= k-1 ... use prefix with a leading zero
            cum = np.concatenate([np.zeros((2, 1, 22), np.int64), np.cumsum(hist, axis=1)], axis=1)
            cum_lat3 = np.concatenate([np.zeros((1, 22)), np.cumsum(lat3, axis=0)], axis=0)
            reach2_count = int(idx.size)
            stage2_latency = float(lat2_by_b2.sum())
            # Stage-2 decisions for all 231 (L2, H2):
            #   real2: b2 <= l2   -> cum[:, l2+1, :].sum
            #   fake2: b2 >= h2+1 -> total - cum[:, h2+1, :].sum
            #   pass2: l2 < b2 <= h2 -> cum[:, h2+1] - cum[:, l2+1]
            tot = cum[:, -1, :].sum(axis=1)  # (2,)
            real2 = cum[:, l2 + 1, :].sum(axis=2)  # (2, 231)
            upto_h2 = cum[:, h2 + 1, :]  # (2, 231, 22)
            fake2 = tot[:, None] - upto_h2.sum(axis=2)
            pass2 = upto_h2 - cum[:, l2 + 1, :]  # (2, 231, 22) by b3
            pass2_lat = cum_lat3[h2 + 1, :] - cum_lat3[l2 + 1, :]  # (231, 22)
            # Stage 3 with T index t: real3 = b3 <= t
            real3 = np.cumsum(pass2, axis=2)[:, :, :21]  # (2, 231, 21)
            pass2_total = pass2.sum(axis=2)  # (2, 231)
            fake3 = pass2_total[:, :, None] - real3
            fake_acc[p, a] = fa1 + real2[1][:, None] + real3[1]
            real_rej[p, a] = fr1 + fake2[0][:, None] + fake3[0]
            reach2[p, a] = reach2_count
            reach3[p, a] = (pass2_total.sum(axis=0))[:, None]
            latency_sum[p, a] = (base_latency + stage2_latency + pass2_lat.sum(axis=1))[:, None]
    return CascadeTable(
        data_name=data.name,
        n_real=data.n_real,
        n_fake=data.n_fake,
        fake_accepted=fake_acc,
        real_rejected=real_rej,
        mean_latency_ms=latency_sum / n,
        reach2=reach2,
        reach3=reach3,
        seconds=time.perf_counter() - started,
    )


def evaluate_cascade(data: Data, stages: list[tuple[int, float, float]]) -> dict:
    """Direct per-sample evaluation of one cascade (for testing / GP checks).

    stages: list of (detector_index, L, H); the last stage uses L as T.
    """
    labels = data.labels
    pred = np.full(len(labels), -1, dtype=np.int8)
    latency = np.zeros(len(labels))
    undecided = np.ones(len(labels), dtype=bool)
    for i, (d, low, high) in enumerate(stages):
        s = data.scores[:, d]
        latency[undecided] += data.latency_ms[undecided, d]
        if i == len(stages) - 1:
            pred[undecided] = (s[undecided] >= low).astype(np.int8)
            undecided[:] = False
            break
        real = undecided & (s < low)
        fake = undecided & (s >= high)
        pred[real] = 0
        pred[fake] = 1
        undecided &= ~(real | fake)
    pred[undecided] = 0  # cannot happen when the last stage is reached
    fa = int(np.count_nonzero((labels == 1) & (pred == 0)))
    fr = int(np.count_nonzero((labels == 0) & (pred == 1)))
    return {
        "fake_accepted": fa,
        "real_rejected": fr,
        "far": fa / data.n_fake,
        "frr": fr / data.n_real,
        "mean_latency_ms": float(latency.mean()),
    }
