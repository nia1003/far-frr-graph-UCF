"""Parallel fusion  q = a*Xception + b*UCF + (1-a-b)*SBI,  Fake iff q >= gamma.

Search procedure requested in the 2026-11-19 meeting:
  1. coarse grid: a, b, gamma in steps of 0.01  (5,151 weight pairs x 101 gammas)
  2. refine: around the coarse optimum take +-10 % (or +-20 %) in every
     coordinate and split it into 100 equal steps (101 points per axis)
Every video's three scores are fixed, so each grid point is only a re-count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from common import DETECTORS, Data


@dataclass
class ParallelGrid:
    """Flat arrays over all evaluated (a, b, gamma) points."""

    a: np.ndarray
    b: np.ndarray
    gamma: np.ndarray
    fake_accepted: np.ndarray
    real_rejected: np.ndarray
    n_real: int
    n_fake: int
    seconds: float
    label: str = ""

    @property
    def far(self) -> np.ndarray:
        return self.fake_accepted / self.n_fake

    @property
    def frr(self) -> np.ndarray:
        return self.real_rejected / self.n_real

    def weights(self, i: int) -> np.ndarray:
        return np.array([self.a[i], self.b[i], 1.0 - self.a[i] - self.b[i]])


def _count(data: Data, weight_pairs: np.ndarray, gammas_for_pair) -> ParallelGrid:
    """Evaluate gamma values for every weight pair with one sort per pair."""
    started = time.perf_counter()
    real_scores_matrix = data.scores[data.labels == 0]
    fake_scores_matrix = data.scores[data.labels == 1]
    A, B, G, FA, FR = [], [], [], [], []
    for k, (a, b) in enumerate(weight_pairs):
        w = np.array([a, b, 1.0 - a - b])
        real_q = np.sort(real_scores_matrix @ w)
        fake_q = np.sort(fake_scores_matrix @ w)
        gammas = gammas_for_pair(k)
        # Fake iff q >= gamma. Fakes accepted as Real: q < gamma.
        fa = np.searchsorted(fake_q, gammas, side="left")
        fr = len(real_q) - np.searchsorted(real_q, gammas, side="left")
        A.append(np.full(len(gammas), a))
        B.append(np.full(len(gammas), b))
        G.append(gammas)
        FA.append(fa)
        FR.append(fr)
    return ParallelGrid(
        a=np.concatenate(A), b=np.concatenate(B), gamma=np.concatenate(G),
        fake_accepted=np.concatenate(FA).astype(np.int32),
        real_rejected=np.concatenate(FR).astype(np.int32),
        n_real=data.n_real, n_fake=data.n_fake,
        seconds=time.perf_counter() - started,
    )


def coarse_grid(data: Data, step: float = 0.01) -> ParallelGrid:
    n = int(round(1 / step))
    pairs = np.array([(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)])
    gammas = np.round(np.arange(n + 1) / n, 10)
    grid = _count(data, pairs, lambda _k: gammas)
    grid.label = f"coarse step={step:g}"
    return grid


def refine(data: Data, center: tuple[float, float, float], radius: float = 0.10,
           steps: int = 100) -> ParallelGrid:
    """+-radius box around (a, b, gamma), `steps` intervals per axis, clipped to
    the feasible region a, b >= 0, a + b <= 1, 0 <= gamma <= 1."""
    a0, b0, g0 = center
    axis = np.linspace(-radius, radius, steps + 1)
    a_vals = a0 + axis
    b_vals = b0 + axis
    g_vals = g0 + axis
    a_vals = a_vals[(a_vals >= -1e-12) & (a_vals <= 1 + 1e-12)].clip(0, 1)
    b_vals = b_vals[(b_vals >= -1e-12) & (b_vals <= 1 + 1e-12)].clip(0, 1)
    g_vals = g_vals[(g_vals >= -1e-12) & (g_vals <= 1 + 1e-12)].clip(0, 1)
    pairs = np.array([(a, b) for a in a_vals for b in b_vals if a + b <= 1 + 1e-12])
    pairs[:, 1] = np.minimum(pairs[:, 1], 1 - pairs[:, 0])
    grid = _count(data, pairs, lambda _k: g_vals)
    grid.label = f"refine +-{radius:g} / {steps}"
    return grid


def evaluate_point(data: Data, w: np.ndarray, gamma: float) -> dict:
    q = data.scores @ w
    pred = q >= gamma
    fa = int(np.count_nonzero((data.labels == 1) & ~pred))
    fr = int(np.count_nonzero((data.labels == 0) & pred))
    used = w > 1e-12
    return {
        "fake_accepted": fa, "real_rejected": fr,
        "far": fa / data.n_fake, "frr": fr / data.n_real,
        # all detectors with non-zero weight must run on every video
        "latency_parallel_ms": float(data.latency_ms[:, used].max(axis=1).mean()),
        "latency_sequential_ms": float(data.latency_ms[:, used].sum(axis=1).mean()),
        "detectors_used": "+".join(d for d, u in zip(DETECTORS, used) if u),
    }
