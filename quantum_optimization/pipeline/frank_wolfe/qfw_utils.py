"""
Utility functions for the Quantum Frank-Wolfe algorithm.

* Step-size schedules.
* Frank-Wolfe duality gap.
* Continuous-objective evaluation.
* Convergence tracker dataclass.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
#  Step-size schedules
# ---------------------------------------------------------------------------

def step_size(t: int, rule: str = "standard") -> float:
    """
    Compute the Frank-Wolfe step size at iteration *t* (0-based).

    Parameters
    ----------
    t : int
        Current iteration index.
    rule : str
        ``'standard'``    –  gamma_t = 2 / (t + 2)  (classic FW)
        ``'diminishing'`` –  gamma_t = 1 / (t + 1)
        ``'constant'``    –  gamma_t = 0.5

    Returns
    -------
    float
        Step size in (0, 1].
    """
    if rule == "standard":
        return 2.0 / (t + 2)
    if rule == "diminishing":
        return 1.0 / (t + 1)
    if rule == "constant":
        return 0.5
    raise ValueError(f"Unknown step-size rule: '{rule}'")


# ---------------------------------------------------------------------------
#  Diagnostics
# ---------------------------------------------------------------------------

def frank_wolfe_gap(grad: np.ndarray, x: np.ndarray, s: np.ndarray) -> float:
    """
    Frank-Wolfe (duality) gap:  g_t = ⟨∇f(x), x − s⟩.

    For convex *f* this is an upper bound on the sub-optimality
    f(x) − f(x*).  A gap of zero implies optimality.
    """
    return float(grad @ (x - s))


def evaluate_continuous_objective(
    Q: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    offset: float = 0.0,
) -> float:
    """Evaluate f(x) = x^T Q x + c^T x + offset."""
    return float(x @ Q @ x + c @ x + offset)


# ---------------------------------------------------------------------------
#  Convergence tracker
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceTracker:
    """Accumulates per-iteration metrics for the Q-FW loop."""

    objective_values: List[float] = field(default_factory=list)
    fw_gaps: List[float] = field(default_factory=list)
    step_sizes: List[float] = field(default_factory=list)
    lmo_objectives: List[float] = field(default_factory=list)
    lmo_times: List[float] = field(default_factory=list)
    vertices: List[np.ndarray] = field(default_factory=list)
    iterates: List[np.ndarray] = field(default_factory=list)

    def record(
        self,
        objective: float,
        fw_gap: float,
        gamma: float,
        lmo_objective: float,
        lmo_time: float,
        vertex: np.ndarray,
        iterate: np.ndarray,
    ):
        """Append one iteration's worth of metrics."""
        self.objective_values.append(objective)
        self.fw_gaps.append(fw_gap)
        self.step_sizes.append(gamma)
        self.lmo_objectives.append(lmo_objective)
        self.lmo_times.append(lmo_time)
        self.vertices.append(vertex.copy())
        self.iterates.append(iterate.copy())

    def is_converged(self, tol: float = 1e-6) -> bool:
        """True when the latest FW gap is below *tol*."""
        if not self.fw_gaps:
            return False
        return self.fw_gaps[-1] < tol

    def summary(self) -> dict:
        """Return a JSON-serialisable summary of the run."""
        return {
            "num_iterations": len(self.objective_values),
            "final_objective": self.objective_values[-1] if self.objective_values else None,
            "final_fw_gap": self.fw_gaps[-1] if self.fw_gaps else None,
            "total_lmo_time": sum(self.lmo_times),
            "objective_history": self.objective_values,
            "fw_gap_history": self.fw_gaps,
            "step_size_history": self.step_sizes,
        }
