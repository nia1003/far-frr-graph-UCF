"""MILP model for parallel fusion of Xception, UCF, and SBI scores."""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import time

import highspy
import numpy as np
from scipy.sparse import coo_matrix


REQUIRED_COLUMNS = {
    "dataset",
    "sample_id",
    "label(ground_truth)",
    "xception_fake_score",
    "ucf_fake_score",
    "sbi_fake_score",
}


HIGHS_TO_SCIPY_STATUS = {
    "Optimal": 0,
    "Time limit reached": 1,
    "Iteration limit reached": 1,
    "Solution limit reached": 1,
    "Infeasible": 2,
    "Unbounded": 3,
}


class MilpSolveError(RuntimeError):
    """Raised when the MILP solver does not return a usable solution."""


@dataclass(frozen=True, slots=True)
class FusionSample:
    dataset: str
    sample_id: str
    label: int
    xception_score: float
    ucf_score: float
    sbi_score: float
    source_row: int


@dataclass(slots=True)
class FusionSeed:
    dataset: str
    alpha_xception: float
    alpha_ucf: float
    alpha_sbi: float
    beta: float
    predictions: np.ndarray
    fusion_scores: np.ndarray
    false_accepts: int
    false_rejections: int
    n_real: int
    n_fake: int
    far: float
    frr: float
    objective: float


@dataclass(slots=True)
class FusionResult:
    dataset: str
    alpha_xception: float
    alpha_ucf: float
    alpha_sbi: float
    beta: float
    predictions: np.ndarray
    fusion_scores: np.ndarray
    false_accepts: int
    false_rejections: int
    n_real: int
    n_fake: int
    far: float
    frr: float
    objective: float
    optimal: bool
    solver_status: int
    solver_message: str
    best_bound: float | None
    mip_gap: float | None
    mip_node_count: int | None
    termination_reason: str
    solution_source: str
    grid_start_objective: float | None
    dominance_cut_count: int = 0
    tight_big_m: bool = False
    solve_seconds: float | None = None


def _termination_reason(status: int, message: str) -> str:
    message_lower = message.lower()
    if status == 0:
        return "optimal"
    if status == 1:
        if "time limit" in message_lower:
            return "time_limit"
        if "iteration" in message_lower or "node limit" in message_lower:
            return "iteration_or_node_limit"
        return "limit_reached"
    if status == 2:
        return "infeasible"
    if status == 3:
        return "unbounded"
    return "solver_error_or_other"


def _parse_score(value: str, column: str, path: Path, row_number: int) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: row {row_number} has invalid {column}: {value!r}"
        ) from error

    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(
            f"{path}: row {row_number} has out-of-range {column}: {score}"
        )
    return score


def read_fusion_csv(path: Path) -> list[FusionSample]:
    """Read and validate a merged score CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Fusion CSV not found: {path}")

    samples: list[FusionSample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            dataset = row["dataset"].strip()
            sample_id = row["sample_id"].strip()
            if not dataset:
                raise ValueError(f"{path}: row {row_number} has a blank dataset")
            if not sample_id:
                raise ValueError(f"{path}: row {row_number} has a blank sample_id")

            try:
                label = int(row["label(ground_truth)"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}: row {row_number} has an invalid ground-truth label"
                ) from error
            if label not in (0, 1):
                raise ValueError(
                    f"{path}: row {row_number} label must be 0 or 1, got {label}"
                )

            samples.append(
                FusionSample(
                    dataset=dataset,
                    sample_id=sample_id,
                    label=label,
                    xception_score=_parse_score(
                        row["xception_fake_score"],
                        "xception_fake_score",
                        path,
                        row_number,
                    ),
                    ucf_score=_parse_score(
                        row["ucf_fake_score"],
                        "ucf_fake_score",
                        path,
                        row_number,
                    ),
                    sbi_score=_parse_score(
                        row["sbi_fake_score"],
                        "sbi_fake_score",
                        path,
                        row_number,
                    ),
                    source_row=row_number,
                )
            )

    if not samples:
        raise ValueError(f"{path} contains no samples")

    datasets = {sample.dataset for sample in samples}
    if len(datasets) != 1:
        raise ValueError(f"{path} contains multiple datasets: {sorted(datasets)}")
    return samples


def _stable_threshold(
    fusion_scores: np.ndarray,
    predictions: np.ndarray,
    epsilon: float,
) -> float:
    """Choose a numerically stable beta from the feasible threshold interval."""
    predicted_real = fusion_scores[predictions == 0]
    predicted_fake = fusion_scores[predictions == 1]

    lower = 0.0 if predicted_real.size == 0 else float(predicted_real.max() + epsilon)
    upper = 1.0 if predicted_fake.size == 0 else float(predicted_fake.min())
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    numerical_tolerance = 1e-12
    if lower > upper + numerical_tolerance:
        raise MilpSolveError(
            "The solver returned predictions that do not admit a valid threshold: "
            f"lower={lower}, upper={upper}"
        )
    if lower > upper:
        # A sub-picounit overlap can arise from floating-point addition of
        # epsilon. Choosing the Fake boundary preserves q == beta -> e_j = 1.
        return upper
    return (lower + upper) / 2.0


def per_sample_big_m(scores: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """Tightest valid Big-M values per sample when beta is in [0, 1].

    Because the weights lie on the simplex, q_j is always inside
    [min_k s_jk, max_k s_jk].  Therefore
      Real row  (e_j = 0 must not bind q_j >= beta - M1): M1_j = 1 - min_k s_jk
      Fake row  (e_j = 1 must not bind q_j <= beta - eps + M2 + eps): M2_j = max_k s_jk
    Both are <= 1, the previous global M.
    """
    lower_m = 1.0 - scores.min(axis=1)
    upper_m = scores.max(axis=1)
    return lower_m, upper_m


def _build_indicator_constraints(
    scores: np.ndarray,
    epsilon: float,
    *,
    big_m: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[coo_matrix, np.ndarray, np.ndarray]:
    """Build strict Big-M rows for e_j = 1 iff q_j >= beta.

    Variable order is alpha_X, alpha_U, alpha_S, beta, e_1, ..., e_N,
    followed by one fixed objective-offset variable.  ``big_m`` optionally
    provides per-sample (M1, M2); the default is the global M = 1.
    """
    sample_count = scores.shape[0]
    if big_m is None:
        lower_m = np.ones(sample_count)
        upper_m = np.ones(sample_count)
    else:
        lower_m, upper_m = big_m
    variable_count = 5 + sample_count
    constraint_count = 1 + 2 * sample_count
    row_indices: list[int] = [0, 0, 0]
    column_indices: list[int] = [0, 1, 2]
    values: list[float] = [1.0, 1.0, 1.0]
    constraint_lower = np.full(constraint_count, -np.inf, dtype=float)
    constraint_upper = np.full(constraint_count, np.inf, dtype=float)
    constraint_lower[0] = 1.0
    constraint_upper[0] = 1.0

    for sample_index, detector_scores in enumerate(scores):
        prediction_column = 4 + sample_index
        lower_row = 1 + 2 * sample_index
        upper_row = lower_row + 1
        m1 = float(lower_m[sample_index])
        m2 = float(upper_m[sample_index])

        for detector_column, score in enumerate(detector_scores):
            row_indices.extend((lower_row, upper_row))
            column_indices.extend((detector_column, detector_column))
            values.extend((float(score), float(score)))

        # q_j - beta >= -M1(1 - e_j)  ->  q_j - beta - M1*e_j >= -M1
        row_indices.extend((lower_row, lower_row))
        column_indices.extend((3, prediction_column))
        values.extend((-1.0, -m1))
        constraint_lower[lower_row] = -m1

        # q_j - beta <= -epsilon + (M2 + epsilon)*e_j
        row_indices.extend((upper_row, upper_row))
        column_indices.extend((3, prediction_column))
        values.extend((-1.0, -(m2 + epsilon)))
        constraint_upper[upper_row] = -epsilon

    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(constraint_count, variable_count),
    ).tocsr()
    return matrix, constraint_lower, constraint_upper


def dominance_pairs(scores: np.ndarray) -> np.ndarray:
    """Return (a, b) pairs with s_a >= s_b componentwise (transitively reduced).

    With non-negative weights q_a >= q_b, so e_b = 1 implies e_a = 1.  Only the
    transitive reduction is returned: if a >= c >= b, the pair (a, b) is
    implied by (a, c) and (c, b) and is dropped.  Identical score vectors are
    ordered by index so that they form a chain instead of a 2-cycle.
    """
    n = scores.shape[0]
    geq = (scores[:, None, :] >= scores[None, :, :]).all(axis=-1)
    equal = (scores[:, None, :] == scores[None, :, :]).all(axis=-1)
    # For identical vectors keep only a < b to avoid cycles.
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    dominates = geq & (~equal | upper)
    np.fill_diagonal(dominates, False)
    as_float = dominates.astype(np.float32)
    implied = (as_float @ as_float) > 0
    reduced = dominates & ~implied
    a, b = np.nonzero(reduced)
    return np.column_stack((a, b))


def exact_beta_grid_seed(
    samples: list[FusionSample],
    *,
    step: float = 0.01,
    epsilon: float = 1e-6,
) -> "FusionSeed":
    """Grid over the weight simplex with the exact best threshold per weight.

    For a fixed weight vector the best threshold is found by sorting the
    fused scores, so beta has no quantization error.  Only cuts whose gap is
    at least ``epsilon`` are considered so the seed satisfies the MILP's
    strict rule.
    """
    labels = np.fromiter((sample.label for sample in samples), dtype=np.int8)
    scores = np.array(
        [(s.xception_score, s.ucf_score, s.sbi_score) for s in samples], dtype=float
    )
    n_real = int(np.count_nonzero(labels == 0))
    n_fake = int(np.count_nonzero(labels == 1))
    intervals = int(round(1.0 / step))
    best: tuple[float, float, np.ndarray, float] | None = None
    for i in range(intervals + 1):
        for j in range(intervals + 1 - i):
            weights = np.array([i, j, intervals - i - j], dtype=float) / intervals
            fused = scores @ weights
            order = np.argsort(fused, kind="stable")
            sorted_q = fused[order]
            sorted_y = labels[order]
            # Cut k: the first k sorted samples are Real, the rest Fake.
            missed_fake = np.concatenate(([0], np.cumsum(sorted_y == 1))) / n_fake
            flagged_real = (
                n_real - np.concatenate(([0], np.cumsum(sorted_y == 0)))
            ) / n_real
            valid = np.ones(len(fused) + 1, dtype=bool)
            valid[1:-1] = sorted_q[1:] - sorted_q[:-1] >= epsilon * 4
            objective = np.where(valid, missed_fake + flagged_real, np.inf)
            k = int(np.argmin(objective))
            balance = abs(missed_fake[k] - flagged_real[k])
            candidate = (float(objective[k]), balance)
            if best is None or candidate < (best[0], best[1]):
                if k == 0:
                    beta = 0.0
                elif k == len(fused):
                    beta = 1.0
                else:
                    beta = float(sorted_q[k])
                best = (candidate[0], candidate[1], weights, beta)
    assert best is not None
    _, _, weights, beta = best
    return build_fusion_seed(
        samples,
        alpha_xception=float(weights[0]),
        alpha_ucf=float(weights[1]),
        alpha_sbi=float(weights[2]),
        beta=beta,
        epsilon=epsilon,
    )


def build_fusion_seed(
    samples: list[FusionSample],
    *,
    alpha_xception: float,
    alpha_ucf: float,
    alpha_sbi: float,
    beta: float,
    epsilon: float = 1e-6,
) -> FusionSeed:
    """Validate a grid-search point and convert it to a strict MILP incumbent."""
    if not samples:
        raise ValueError("At least one sample is required")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be strictly between 0 and 1")

    datasets = {sample.dataset for sample in samples}
    if len(datasets) != 1:
        raise ValueError(f"Expected one dataset, got: {sorted(datasets)}")
    dataset = next(iter(datasets))

    weights = np.array(
        [alpha_xception, alpha_ucf, alpha_sbi], dtype=float
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValueError(f"Grid-start weights must be in [0, 1], got {weights}")
    if not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"Grid-start weights must sum to 1, got {weights}")
    if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError(f"Grid-start beta must be in [0, 1], got {beta}")

    labels = np.fromiter((sample.label for sample in samples), dtype=np.int8)
    scores = np.array(
        [
            (sample.xception_score, sample.ucf_score, sample.sbi_score)
            for sample in samples
        ],
        dtype=float,
    )
    n_real = int(np.count_nonzero(labels == 0))
    n_fake = int(np.count_nonzero(labels == 1))
    if n_real == 0 or n_fake == 0:
        raise ValueError(
            f"{dataset} needs both classes; n_real={n_real}, n_fake={n_fake}"
        )

    fusion_scores = scores @ weights
    predictions = (fusion_scores >= beta).astype(np.int8)
    try:
        stable_beta = _stable_threshold(fusion_scores, predictions, epsilon)
    except MilpSolveError as error:
        raise ValueError(
            f"Grid-start point for {dataset} is not feasible under epsilon={epsilon}: "
            f"{error}"
        ) from error
    stable_predictions = (fusion_scores >= stable_beta).astype(np.int8)
    if not np.array_equal(predictions, stable_predictions):
        raise ValueError(
            f"Grid-start point for {dataset} changes labels under the strict epsilon rule"
        )

    false_accepts = int(np.count_nonzero((labels == 0) & (predictions == 1)))
    false_rejections = int(np.count_nonzero((labels == 1) & (predictions == 0)))
    far = false_accepts / n_real
    frr = false_rejections / n_fake
    return FusionSeed(
        dataset=dataset,
        alpha_xception=float(weights[0]),
        alpha_ucf=float(weights[1]),
        alpha_sbi=float(weights[2]),
        beta=stable_beta,
        predictions=predictions,
        fusion_scores=fusion_scores,
        false_accepts=false_accepts,
        false_rejections=false_rejections,
        n_real=n_real,
        n_fake=n_fake,
        far=far,
        frr=frr,
        objective=far + frr,
    )


def solve_parallel_fusion(
    samples: list[FusionSample],
    *,
    epsilon: float = 1e-6,
    time_limit: float | None = None,
    mip_rel_gap: float | None = None,
    display_solver_log: bool = False,
    grid_seed: FusionSeed | None = None,
    tight_big_m: bool = True,
    dominance_cuts: bool = True,
    threads: int | None = None,
    random_seed: int = 0,
) -> FusionResult:
    """Solve the parallel-fusion MILP for one dataset with HiGHS (highspy).

    Improvements over the original scipy formulation:
      * ``grid_seed`` is passed to HiGHS as a real MIP start (setSolution),
        instead of an objective-cutoff row.
      * ``tight_big_m`` uses per-sample Big-M values (see per_sample_big_m).
      * ``dominance_cuts`` adds e_a >= e_b for every transitively reduced
        dominance pair (see dominance_pairs).
    """
    if not samples:
        raise ValueError("At least one sample is required")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be strictly between 0 and 1")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if mip_rel_gap is not None and not 0.0 <= mip_rel_gap <= 1.0:
        raise ValueError("mip_rel_gap must be between 0 and 1")

    datasets = {sample.dataset for sample in samples}
    if len(datasets) != 1:
        raise ValueError(f"Expected one dataset, got: {sorted(datasets)}")
    dataset = next(iter(datasets))
    if grid_seed is not None:
        if grid_seed.dataset != dataset:
            raise ValueError(
                f"Grid seed is for {grid_seed.dataset!r}, expected {dataset!r}"
            )
        if len(grid_seed.predictions) != len(samples):
            raise ValueError("Grid seed sample count does not match MILP input")

    labels = np.fromiter((sample.label for sample in samples), dtype=np.int8)
    scores = np.array(
        [
            (sample.xception_score, sample.ucf_score, sample.sbi_score)
            for sample in samples
        ],
        dtype=float,
    )
    sample_count = len(samples)
    n_real = int(np.count_nonzero(labels == 0))
    n_fake = int(np.count_nonzero(labels == 1))
    if n_real == 0 or n_fake == 0:
        raise ValueError(
            f"{dataset} needs both classes; n_real={n_real}, n_fake={n_fake}"
        )

    # Variable order: alpha_X, alpha_U, alpha_S, beta, e_1, ..., e_N, followed
    # by a fixed-one variable that carries the constant term of FRR, so the
    # solver objective and reported MIP gap equal FAR + FRR.
    prediction_start = 4
    objective_offset_index = prediction_start + sample_count
    variable_count = objective_offset_index + 1
    objective_coefficients = np.zeros(variable_count, dtype=float)
    prediction_objective = objective_coefficients[
        prediction_start:objective_offset_index
    ]
    prediction_objective[labels == 0] = 1.0 / n_real
    prediction_objective[labels == 1] = -1.0 / n_fake
    objective_coefficients[objective_offset_index] = 1.0

    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.ones(variable_count, dtype=float)
    lower_bounds[objective_offset_index] = 1.0

    big_m = per_sample_big_m(scores, epsilon) if tight_big_m else None
    matrix, constraint_lower, constraint_upper = _build_indicator_constraints(
        scores, epsilon, big_m=big_m
    )

    pairs = dominance_pairs(scores) if dominance_cuts else np.empty((0, 2), int)

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", bool(display_solver_log))
    # HiGHS' default MIP feasibility tolerance is too close to epsilon=1e-6
    # and can numerically reintroduce the boundary ambiguity.
    feasibility_tolerance = max(1e-10, min(1e-8, epsilon / 100.0))
    highs.setOptionValue("mip_feasibility_tolerance", feasibility_tolerance)
    highs.setOptionValue("primal_feasibility_tolerance", feasibility_tolerance)
    highs.setOptionValue("random_seed", int(random_seed))
    if threads is not None:
        highs.setOptionValue("threads", int(threads))
    if time_limit is not None:
        highs.setOptionValue("time_limit", float(time_limit))
    if mip_rel_gap is not None:
        highs.setOptionValue("mip_rel_gap", float(mip_rel_gap))

    highs.addVars(variable_count, lower_bounds, upper_bounds)
    highs.changeColsCost(
        variable_count, np.arange(variable_count, dtype=np.int32), objective_coefficients
    )
    integer_columns = np.arange(prediction_start, objective_offset_index, dtype=np.int32)
    highs.changeColsIntegrality(
        len(integer_columns),
        integer_columns,
        np.full(len(integer_columns), highspy.HighsVarType.kInteger),
    )
    csr = matrix.tocsr()
    highs.addRows(
        csr.shape[0],
        constraint_lower,
        constraint_upper,
        csr.nnz,
        csr.indptr[:-1].astype(np.int32),
        csr.indices.astype(np.int32),
        csr.data,
    )
    if len(pairs):
        # e_a - e_b >= 0
        pair_count = len(pairs)
        starts = np.arange(0, 2 * pair_count, 2, dtype=np.int32)
        indices = np.empty(2 * pair_count, dtype=np.int32)
        indices[0::2] = prediction_start + pairs[:, 0]
        indices[1::2] = prediction_start + pairs[:, 1]
        values = np.tile([1.0, -1.0], pair_count)
        highs.addRows(
            pair_count,
            np.zeros(pair_count),
            np.full(pair_count, highspy.kHighsInf),
            2 * pair_count,
            starts,
            indices,
            values,
        )

    if grid_seed is not None:
        start = np.zeros(variable_count, dtype=float)
        start[:3] = (grid_seed.alpha_xception, grid_seed.alpha_ucf, grid_seed.alpha_sbi)
        start[3] = grid_seed.beta
        start[prediction_start:objective_offset_index] = grid_seed.predictions
        start[objective_offset_index] = 1.0
        solution_start = highspy.HighsSolution()
        solution_start.col_value = start.tolist()
        solution_start.value_valid = True
        highs.setSolution(solution_start)

    started = time.perf_counter()
    highs.run()
    solve_seconds = time.perf_counter() - started

    model_status = highs.getModelStatus()
    message = highs.modelStatusToString(model_status)
    status = HIGHS_TO_SCIPY_STATUS.get(message, 4)
    info = highs.getInfo()
    has_solution = info.primal_solution_status == 2
    best_bound = float(info.mip_dual_bound)
    best_bound = best_bound if np.isfinite(best_bound) else None
    node_count = int(info.mip_node_count)
    termination_reason = _termination_reason(status, message)

    if not has_solution:
        if grid_seed is None:
            raise MilpSolveError(
                f"MILP failed for {dataset}: status={status}, message={message}"
            )
        solution_x = None
    else:
        solution_x = np.asarray(highs.getSolution().col_value, dtype=float)

    extra = dict(
        dominance_cut_count=int(len(pairs)),
        tight_big_m=bool(tight_big_m),
        solve_seconds=solve_seconds,
    )

    candidate = None
    if solution_x is not None:
        weights = solution_x[:3].copy()
        raw_predictions = solution_x[prediction_start:objective_offset_index]
        predictions = np.rint(raw_predictions).astype(np.int8)
        if np.max(np.abs(raw_predictions - predictions)) > 1e-5:
            raise MilpSolveError("The MILP returned non-integral prediction variables")
        if abs(float(weights.sum()) - 1.0) > 1e-6:
            raise MilpSolveError(f"Detector weights do not sum to 1: {weights}")
        fusion_scores = scores @ weights
        beta = _stable_threshold(fusion_scores, predictions, epsilon)
        threshold_predictions = (fusion_scores >= beta).astype(np.int8)
        if not np.array_equal(predictions, threshold_predictions):
            mismatch_count = int(np.count_nonzero(predictions != threshold_predictions))
            raise MilpSolveError(
                f"MILP predictions disagree with q >= beta for {mismatch_count} samples"
            )
        false_accepts = int(np.count_nonzero((labels == 0) & (predictions == 1)))
        false_rejections = int(np.count_nonzero((labels == 1) & (predictions == 0)))
        candidate = (weights, beta, predictions, fusion_scores,
                     false_accepts, false_rejections)

    use_seed = candidate is None or (
        grid_seed is not None
        and grid_seed.objective
        < candidate[4] / n_real + candidate[5] / n_fake - 1e-12
    )
    optimal = termination_reason == "optimal"
    if use_seed:
        assert grid_seed is not None
        gap = None
        if best_bound is not None:
            gap = max(0.0, (grid_seed.objective - best_bound)
                      / max(abs(grid_seed.objective), 1e-12))
        return FusionResult(
            dataset=dataset,
            alpha_xception=grid_seed.alpha_xception,
            alpha_ucf=grid_seed.alpha_ucf,
            alpha_sbi=grid_seed.alpha_sbi,
            beta=grid_seed.beta,
            predictions=grid_seed.predictions,
            fusion_scores=grid_seed.fusion_scores,
            false_accepts=grid_seed.false_accepts,
            false_rejections=grid_seed.false_rejections,
            n_real=grid_seed.n_real,
            n_fake=grid_seed.n_fake,
            far=grid_seed.far,
            frr=grid_seed.frr,
            objective=grid_seed.objective,
            optimal=optimal and gap is not None and gap <= 1e-9,
            solver_status=status,
            solver_message=f"{message}. Using validated grid seed.",
            best_bound=best_bound,
            mip_gap=gap,
            mip_node_count=node_count,
            termination_reason=termination_reason,
            solution_source="grid_seed",
            grid_start_objective=grid_seed.objective,
            **extra,
        )

    assert candidate is not None
    weights, beta, predictions, fusion_scores, false_accepts, false_rejections = candidate
    far = false_accepts / n_real
    frr = false_rejections / n_fake
    objective = far + frr
    gap = None
    if best_bound is not None:
        gap = max(0.0, (objective - best_bound) / max(abs(objective), 1e-12))
    return FusionResult(
        dataset=dataset,
        alpha_xception=float(weights[0]),
        alpha_ucf=float(weights[1]),
        alpha_sbi=float(weights[2]),
        beta=beta,
        predictions=predictions,
        fusion_scores=fusion_scores,
        false_accepts=false_accepts,
        false_rejections=false_rejections,
        n_real=n_real,
        n_fake=n_fake,
        far=far,
        frr=frr,
        objective=objective,
        optimal=optimal,
        solver_status=status,
        solver_message=message,
        best_bound=best_bound,
        mip_gap=gap,
        mip_node_count=node_count,
        termination_reason=termination_reason,
        solution_source="milp",
        grid_start_objective=None if grid_seed is None else grid_seed.objective,
        **extra,
    )
