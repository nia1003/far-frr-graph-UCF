import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cascade import GRID, ORDERINGS, PAIRS, enumerate_cascade, evaluate_cascade  # noqa: E402
from common import Data  # noqa: E402


def make_data(n=300, seed=1):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.5).astype(np.int8)
    scores = np.clip(rng.random((n, 3)) * 0.8 + 0.2 * labels[:, None], 0, 1)
    scores[:5] = GRID[[3, 7, 20, 0, 10]][:, None]  # exact grid values
    lat = np.tile([100.0, 200.0, 1000.0], (n, 1))
    return Data("toy", [str(i) for i in range(n)], ["toy"] * n, labels, scores, lat)


def test_enumeration_matches_direct_evaluation():
    data = make_data()
    table = enumerate_cascade(data)
    rng = np.random.default_rng(0)
    for _ in range(400):
        p = rng.integers(6)
        a = rng.integers(len(PAIRS))
        b = rng.integers(len(PAIRS))
        t = rng.integers(21)
        order = ORDERINGS[p]
        (l1, h1), (l2, h2) = PAIRS[a], PAIRS[b]
        stages = [(order[0], GRID[l1], GRID[h1]), (order[1], GRID[l2], GRID[h2]),
                  (order[2], GRID[t], GRID[t])]
        expected = evaluate_cascade(data, stages)
        assert table.fake_accepted[p, a, b, t] == expected["fake_accepted"]
        assert table.real_rejected[p, a, b, t] == expected["real_rejected"]
        assert np.isclose(table.mean_latency_ms[p, a, b, t], expected["mean_latency_ms"])
