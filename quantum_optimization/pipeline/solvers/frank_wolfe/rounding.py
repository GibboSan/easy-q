"""Section-4 rounding for FWAL outputs.

Given an approximate lifted solution W, this module follows the paper's
decoding path:

1. Extract X_hat from the original-variable block of W (ignoring the
   homogeneous coordinate and any slack-variable columns).
2. Compute the best rank-1 approximation  x_hat x_hat^T  of X_hat using
   the principal eigenvector.
3. Project x_hat onto the original feasible set using constraint-aware
   rounding (sum-1 argmax selection + 0.5-threshold for the rest).
"""

from typing import Dict

import numpy as np

from pipeline.problems.abstract_problem import AbstractProblem


def _to_bitstring(x_binary: np.ndarray) -> str:
    return "".join(str(int(v)) for v in x_binary)


def _best_rank_one_factor(X: np.ndarray) -> np.ndarray:
    """Return x such that x x^T best approximates X in Frobenius norm.

    Uses the principal eigenvector of the symmetric part of X, scaled by
    the square root of the corresponding eigenvalue and clipped to [0, 1].
    """
    eigvals, eigvecs = np.linalg.eigh(X)
    idx = np.argmax(eigvals)
    sigma = max(float(eigvals[idx]), 0.0)
    u = np.abs(eigvecs[:, idx])
    x = np.sqrt(sigma) * u
    return np.clip(x, 0.0, 1.0)


def _project_with_constraints(
    x: np.ndarray, problem: AbstractProblem
) -> np.ndarray:
    """Project continuous x to a binary vector respecting sum-1 constraints.

    For each sum-1 constraint group, the variable with the highest
    continuous value is set to 1 and the rest to 0.  Remaining variables
    are rounded at the 0.5 threshold.
    """
    x_proj = (x >= 0.5).astype(float)

    if getattr(problem, "constraints_sum_1", None) is not None:
        for constraint in problem.constraints_sum_1:
            indices = list(constraint.linear.to_dict().keys())
            best_idx = max(indices, key=lambda idx: x[idx])
            for idx in indices:
                x_proj[idx] = 1.0 if idx == best_idx else 0.0

    return x_proj


def round_from_W(
    W: np.ndarray,
    problem: AbstractProblem,
    n_original: int,
    *,
    project: bool = True,
) -> Dict:
    """Decode a binary solution from a lifted FWAL iterate W.

    Parameters
    ----------
    W : np.ndarray
        Lifted matrix (p x p).
    problem : AbstractProblem
        The original optimisation problem (for feasibility / cost).
    n_original : int
        Number of original binary variables (excluding slack bits).
    project : bool
        Whether to apply constraint-aware projection.

    Returns
    -------
    dict
        ``x_hat``       - continuous relaxation (before rounding)
        ``X_hat``       - original-variable block of W
        ``bitstring``   - rounded binary string
        ``objective``   - cost of the rounded solution
        ``is_feasible`` - whether the rounded solution is feasible
    """
    X_hat = W[1 : n_original + 1, 1 : n_original + 1]
    x_hat = _best_rank_one_factor(X_hat)

    if project:
        x_bin = _project_with_constraints(x_hat, problem)
    else:
        x_bin = (x_hat >= 0.5).astype(float)

    bitstring = _to_bitstring(x_bin)
    is_feasible = problem.is_feasible(bitstring)[0]

    return {
        "x_hat": x_hat,
        "X_hat": X_hat,
        "bitstring": bitstring,
        "objective": float(problem.evaluate_cost(bitstring)),
        "is_feasible": is_feasible,
    }
