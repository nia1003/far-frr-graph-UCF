#!/usr/bin/env python3
"""Figures for the cascade vs parallel study (results/figures/)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import pandas as pd

from common import RESULTS_DIR

SETS = ["FaceForensics++", "Celeb-DF-v2", "DFDC", "balanced_pooled", "balanced_equal_source"]
C_CASCADE, C_PAR, C_SEQ = "#2a6fdb", "#d9822b", "#8a8a8a"


def main() -> None:
    out = RESULTS_DIR / "figures"
    out.mkdir(exist_ok=True)
    fr = pd.read_csv(RESULTS_DIR / "far_frr_frontier.csv")
    lat = pd.read_csv(RESULTS_DIR / "latency_frontier.csv")

    fig, axes = plt.subplots(1, len(SETS), figsize=(4 * len(SETS), 3.6), sharey=True)
    for ax, s in zip(axes, SETS):
        d = fr[fr.test_set == s]
        ax.step(d.far_limit, d.cascade_min_frr, where="post", color=C_CASCADE, label="cascade (5% grid)")
        ax.step(d.far_limit, d.parallel_min_frr, where="post", color=C_PAR, label="parallel (1% grid)")
        ax.set_title(s, fontsize=10)
        ax.set_xlabel("FAR limit (fake accepted)")
        ax.set_xlim(0, 0.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("min FRR (real rejected)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "far_frr_frontier.png", dpi=150)

    fig, axes = plt.subplots(1, len(SETS), figsize=(4 * len(SETS), 3.6))
    for ax, s in zip(axes, SETS):
        d = lat[lat.test_set == s]
        for arch, color in [("cascade", C_CASCADE), ("parallel (parallel run)", C_PAR),
                            ("parallel (sequential run)", C_SEQ)]:
            e = d[d.architecture == arch]
            ax.step(e.latency_ms, e.min_far_plus_frr, where="post", color=color, label=arch)
        ax.set_xscale("log")
        ax.set_xticks([100, 150, 200, 300, 500, 1000, 2000, 3000])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(s, fontsize=10)
        ax.set_xlim(d.latency_ms.min() * 0.9, d.latency_ms.max() * 1.1)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("mean latency per video (ms)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("min FAR + FRR")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "latency_frontier.png", dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
