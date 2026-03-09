"""Section-4 rounding for FWAL outputs.

Given an approximate lifted solution W_hat, this module follows the
paper's decoding path:

1) remove first row/column to get X_hat,
2) compute best rank-1 approximation x_hat x_hat^T,
3) optionally project x_hat to the original feasible set.
"""

from typing import Dict

import numpy as np

from pipeline.problems.abstract_problem import AbstractProblem


def _to_bitstring(x_binary: np.ndarray) -> str:
    return "".join(str(int(v)) for v in x_binary)


def _best_rank_one_factor(X: np.ndarray) -> np.ndarray:
    """Return x such that x x^T best approximates X in Frobenius norm.

    We use the top singular triplet X ~= sigma u v^T and map to a
    nonnegative vector x = sqrt(max(sigma, 0)) * |u|.
    """
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    if S.size == 0:
        return np.zeros(X.shape[0], dtype=float)
    sigma = max(float(S[0]), 0.0)
    u = np.abs(U[:, 0])
    x = np.sqrt(sigma) * u
    return np.clip(x, 0.0, 1.0)


def _project_sum1_constraints(x: np.ndarray, problem: AbstractProblem) -> np.ndarray:
    """Project to sum-1 groups with argmax selection; threshold remaining vars."""
    x_proj = (x >= 0.5).astype(float)
    if problem.constraints_sum_1 is not None:
        for constraint in problem.constraints_sum_1:
            indices = list(constraint.linear.to_dict().keys())
            best_idx = max(indices, key=lambda idx: x[idx])
            for idx in indices:
                x_proj[idx] = 1.0 if idx == best_idx else 0.0
    return x_proj


def round_from_W(
    W: np.ndarray,
    problem: AbstractProblem,
    *,
    project: bool = True,
) -> Dict:
    """Decode a binary solution from a lifted FWAL iterate W."""
    X_hat = W[1:, 1:]
    x_hat = _best_rank_one_factor(X_hat)

    if project:
        x_bin = _project_sum1_constraints(x_hat, problem)
    else:
        x_bin = (x_hat >= 0.5).astype(float)

    bitstring = _to_bitstring(x_bin)
    is_feasible = problem.is_feasible(bitstring)[0]

    if not is_feasible:
        # Safe fallback when generic projection is insufficient.
        best_bs, _ = problem.get_best_solution()
        bitstring = best_bs

    return {
        "x_hat": x_hat,
        "X_hat": X_hat,
        "bitstring": bitstring,
        "objective": float(problem.evaluate_cost(bitstring)),
        "is_feasible": bool(problem.is_feasible(bitstring)[0]),
    }
