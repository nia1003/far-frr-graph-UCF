# Parallel Fusion MILP

This directory contains the merged detector scores and a mixed-integer linear
program that optimizes each dataset independently.

## Model

The formulation uses the three detector weights, the overall threshold, and
one binary prediction `e_j` per sample. The detector weights are constrained
to sum to one. The objective is:

```text
minimize FAR + FRR
```

`q_j` is substituted directly into the Big-M constraints, so it does not need
to be stored as a separate solver variable. `M = 1` is valid because all input
scores, weights, and the threshold are in `[0, 1]`.

Each sample uses both strict Big-M indicator rows, with `M = 1` and a small
configurable `epsilon` (default `1e-6`):

```text
q_j - beta >= -M(1 - e_j)
q_j - beta <= -epsilon + (M + epsilon)e_j
```

Therefore there is no ambiguous band in which the solver can choose whichever
label improves the objective:

```text
q_j >= beta  -> Fake (1)
q_j <= beta - epsilon -> Real (0)
beta - epsilon < q_j < beta -> infeasible
```

The epsilon-sized infeasible interval is the finite-precision replacement for
the strict inequality `q_j < beta`; it is not a region where either label is
allowed.

## Run

Install the dependencies if necessary:

```powershell
python -m pip install -r parallel-fusion/milp/requirements.txt   # numpy, scipy, highspy
```

Run all datasets from the project root:

```powershell
python parallel-fusion/milp/run_milp.py
```

Run only one dataset:

```powershell
python parallel-fusion/milp/run_milp.py --dataset DFDC
```

Useful optional arguments:

```text
--epsilon 1e-6
--time-limit SECONDS
--mip-rel-gap VALUE
--seed-step 0.01
--no-seed / --no-tight-big-m / --no-dominance   (ablation)
--threads N
--solver-log
```

## Solver improvements (HiGHS via `highspy`)

The solver now calls HiGHS directly through `highspy` instead of
`scipy.optimize.milp`. Four changes, all on by default:

1. **Real MIP start.** Before solving, `exact_beta_grid_seed` searches the
   weight simplex in steps of `--seed-step` (default 0.01) and, for each
   weight vector, finds the best threshold exactly by sorting the fused
   scores (no beta quantisation). That point is passed to HiGHS with
   `setSolution()`, so the MILP can never end worse than this grid. The old
   objective-cutoff row was removed. `--grid-start <csv>` still works and
   replaces the built-in seed. `--no-seed` disables the start.
2. **Per-sample Big-M.** On the simplex, `q_j` is always inside
   `[min_k s_jk, max_k s_jk]`, so `M1_j = 1 - min_k s_jk` and
   `M2_j = max_k s_jk` are valid and never larger than the old global `M = 1`.
   Disable with `--no-tight-big-m`.
3. **Dominance cuts.** If video `a` scores at least as high as video `b` on
   all three detectors, then `q_a >= q_b` for any non-negative weights, so
   `e_b = 1` forces `e_a = 1`. The rows `e_a - e_b >= 0` are added for the
   transitive reduction of the dominance relation (DFDC: 6.48 M dominance
   pairs reduce to 88,842 rows). Disable with `--no-dominance`.
4. **Status is always recorded.** `milp_summary.csv` now has `optimal`,
   `best_bound`, `mip_gap`, `node_count`, `seed_source`, `tight_big_m`,
   `dominance_cut_count` and `solve_seconds` for every dataset. Running one
   dataset merges into the existing summary instead of erasing the other
   rows.

Merging identical score vectors was also considered, but none of the three
datasets contains duplicate score vectors, so it is not implemented.

```powershell
python parallel-fusion/milp/run_milp.py --time-limit 1200
```

Results are written under `parallel-fusion/milp/milp-results/`:

- `milp_summary.csv`: weights, threshold, FAR, FRR, and solver status.
- `<dataset>_milp_predictions.csv`: per-sample fusion score, prediction, and
  error type.

The optimizer diagnostics are printed to the terminal and stored in
`milp_summary.csv` as `best_bound`, `mip_gap`, `node_count`, and
`termination_reason`.

## Rebuild merged inputs

```powershell
python parallel-fusion/merge_scores.py
```
