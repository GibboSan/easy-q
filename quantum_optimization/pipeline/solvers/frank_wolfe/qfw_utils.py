"""FWAL schedules, diagnostics, and convergence tracking."""

from dataclasses import dataclass, field
from typing import List

import numpy as np


def primal_step_size(t: int) -> float:
    """FW primal step size eta_t = 2 / (t + 1) with t starting at 1."""
    return 2.0 / (t + 1)


def penalty_parameter(beta0: float, t: int) -> float:
    """FWAL penalty schedule beta_t = beta0 * sqrt(t + 1)."""
    return float(beta0 * np.sqrt(t + 1.0))


def dual_step_size(beta0: float, rule: str = "constant") -> float:
    """Dual step-size schedule gamma_t.

    Only the constant rule is currently implemented because it is the
    practical choice reported in the paper.
    """
    if rule != "constant":
        raise NotImplementedError("Only constant dual step size is implemented")
    return float(beta0)


def fw_gap_matrix(G: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    """Frank-Wolfe gap in lifted space: <G, W - H>."""
    return float(np.sum(G * (W - H)))


def residual_norm(residual: np.ndarray) -> float:
    """Constraint residual norm ||A(W)-v||_2."""
    return float(np.linalg.norm(residual, ord=2))


@dataclass
class ConvergenceTracker:
    """Accumulates per-iteration FWAL metrics."""

    cp_objective_values: List[float] = field(default_factory=list)
    fw_gaps: List[float] = field(default_factory=list)
    residual_norms: List[float] = field(default_factory=list)
    eta_values: List[float] = field(default_factory=list)
    beta_values: List[float] = field(default_factory=list)
    gamma_values: List[float] = field(default_factory=list)
    lmo_objectives: List[float] = field(default_factory=list)
    lmo_times: List[float] = field(default_factory=list)
    lmo_bitstrings: List[str] = field(default_factory=list)

    def record(
        self,
        cp_objective: float,
        fw_gap: float,
        residual: float,
        eta: float,
        beta: float,
        gamma: float,
        lmo_objective: float,
        lmo_time: float,
        lmo_bitstring: str,
    ) -> None:
        self.cp_objective_values.append(float(cp_objective))
        self.fw_gaps.append(float(fw_gap))
        self.residual_norms.append(float(residual))
        self.eta_values.append(float(eta))
        self.beta_values.append(float(beta))
        self.gamma_values.append(float(gamma))
        self.lmo_objectives.append(float(lmo_objective))
        self.lmo_times.append(float(lmo_time))
        self.lmo_bitstrings.append(lmo_bitstring)

    def summary(self) -> dict:
        return {
            "num_iterations": len(self.cp_objective_values),
            "final_cp_objective": self.cp_objective_values[-1] if self.cp_objective_values else None,
            "final_fw_gap": self.fw_gaps[-1] if self.fw_gaps else None,
            "final_residual_norm": self.residual_norms[-1] if self.residual_norms else None,
            "total_lmo_time": float(sum(self.lmo_times)),
            "cp_objective_history": self.cp_objective_values,
            "fw_gap_history": self.fw_gaps,
            "residual_history": self.residual_norms,
            "eta_history": self.eta_values,
            "beta_history": self.beta_values,
            "gamma_history": self.gamma_values,
        }
