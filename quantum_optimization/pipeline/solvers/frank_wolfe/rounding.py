"""
Rounding strategies for converting a continuous relaxed solution
back to a binary feasible solution.

Three methods are provided:

* **threshold** – deterministic, simple ≥ 0.5 rounding.
* **randomized** – probabilistic Bernoulli sampling (multiple trials,
  best feasible kept).
* **greedy** – constraint-aware: for each sum-1 group choose the
  variable with the highest continuous value; remaining variables
  are threshold-rounded.
"""

from typing import Tuple

import numpy as np

from pipeline.problems.abstract_problem import AbstractProblem


# ---------------------------------------------------------------------------
#  Elementary rounding primitives
# ---------------------------------------------------------------------------

def threshold_round(x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Round each component: 1 if x_i >= *threshold*, else 0."""
    return (x >= threshold).astype(float)


def randomized_round(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample x_i ~ Bernoulli(x_i) independently."""
    return (rng.random(len(x)) < x).astype(float)


def greedy_constraint_round(
    x: np.ndarray,
    problem: AbstractProblem,
) -> np.ndarray:
    """
    Constraint-aware greedy rounding.

    For each sum-1 constraint group, select the variable with the
    highest continuous value and set it to 1 (others in the group to 0).
    Remaining unconstrained variables are threshold-rounded at 0.5.
    """
    result = threshold_round(x.copy())

    if problem.constraints_sum_1 is not None:
        for constraint in problem.constraints_sum_1:
            indices = list(constraint.linear.to_dict().keys())
            best_idx = max(indices, key=lambda i: x[i])
            for idx in indices:
                result[idx] = 1.0 if idx == best_idx else 0.0

    return result


# ---------------------------------------------------------------------------
#  Convenience helpers
# ---------------------------------------------------------------------------

def _to_bitstring(x_binary: np.ndarray) -> str:
    return "".join(str(int(b)) for b in x_binary)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def round_solution(
    x: np.ndarray,
    problem: AbstractProblem,
    method: str = "greedy",
    seed: int = 42,
    num_random_trials: int = 100,
) -> Tuple[str, float]:
    """
    Round a continuous solution to a binary feasible solution.

    Parameters
    ----------
    x : np.ndarray
        Continuous solution in [0, 1]^n.
    problem : AbstractProblem
        Problem instance (feasibility check + cost evaluation).
    method : str
        ``'threshold'``, ``'randomized'``, or ``'greedy'``.
    seed : int
        Seed for the randomised variant.
    num_random_trials : int
        Number of independent Bernoulli trials (randomised only).

    Returns
    -------
    bitstring : str
        Best binary solution found.
    cost : float
        Objective value of that solution.
    """
    if method == "threshold":
        x_bin = threshold_round(x)
        bs = _to_bitstring(x_bin)
        return bs, problem.evaluate_cost(bs)

    if method == "greedy":
        x_bin = greedy_constraint_round(x, problem)
        bs = _to_bitstring(x_bin)
        return bs, problem.evaluate_cost(bs)

    if method == "randomized":
        rng = np.random.default_rng(seed)
        best_bs: str | None = None
        best_cost = float("inf")

        for _ in range(num_random_trials):
            x_bin = randomized_round(x, rng)
            bs = _to_bitstring(x_bin)
            if problem.is_feasible(bs)[0]:
                cost = problem.evaluate_cost(bs)
                if cost < best_cost:
                    best_cost = cost
                    best_bs = bs

        if best_bs is None:
            # Fallback to greedy if no feasible sample was drawn
            return round_solution(x, problem, method="greedy")

        return best_bs, best_cost

    raise ValueError(
        f"Unknown rounding method: '{method}'. "
        "Choose from 'threshold', 'randomized', or 'greedy'."
    )
