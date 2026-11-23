#!/usr/bin/env python3
"""Search-space size of cascade vs parallel fusion as the number of detectors m grows.

cascade : m! orderings x (G(G+1)/2)^(m-1) dual-threshold pairs x G last thresholds
          (shorter cascades are included: a stage with L == H ends the cascade)
          decision variables: 2(m-1) + 1 = 2m - 1 thresholds (+ the ordering)
parallel: weights on the simplex with step 1/W: C(W+m-1, m-1) x (W+1) thresholds
          decision variables: (m-1) free weights + 1 threshold = m
          MILP (exact): m + 1 continuous + N binaries, size grows with videos, not m!
"""

from __future__ import annotations

from math import comb, factorial

from common import RESULTS_DIR, write_csv

G = 21   # 5 % threshold grid
W = 100  # 1 % weight grid


def main() -> None:
    rows = []
    for m in range(1, 7):
        cascade = factorial(m) * (G * (G + 1) // 2) ** (m - 1) * G
        parallel = comb(W + m - 1, m - 1) * (W + 1)
        rows.append({
            "m_detectors": m,
            "cascade_decision_variables": 2 * m - 1,
            "cascade_configs_5pct": cascade,
            "parallel_decision_variables": m,
            "parallel_configs_1pct": parallel,
            "parallel_exact_beta_weight_points_1pct": comb(W + m - 1, m - 1),
            "ratio_cascade_over_parallel": cascade / parallel,
        })
    write_csv(RESULTS_DIR / "complexity.csv", rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
