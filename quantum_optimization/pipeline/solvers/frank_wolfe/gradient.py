"""Copositive program (CP) model construction and FWAL gradient computation.

Implements the lifted model from Yurtsever et al. (2022), Sections 3--4:

    min_W  Tr(C W)    s.t.  A(W) = v,   W in Delta^p

where W is a (p x p) completely positive matrix, C encodes the objective,
and (A, v) encode all affine constraints including:

  - original problem constraints (equality and inequality via slack variables),
  - diag(X) = x,
  - W[0,0] = 1.

Inequality constraints (<= and >=) are converted to equalities by
introducing binary slack variables, expanding the variable space from *n*
to *n_expanded* as described in the Q-FWAL notes.

All constraint matrices M_k are stored in symmetric form so that the
adjoint A*(y) = sum_k y_k M_k is always symmetric, eliminating the need
for post-hoc symmetrisation of the gradient.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from qiskit_optimization.problems import LinearConstraint

from pipeline.problems.abstract_problem import AbstractProblem

logger = logging.getLogger("pipeline_logger")


# ------------------------------------------------------------------ #
#   Data structures
# ------------------------------------------------------------------ #


@dataclass
class SlackInfo:
    """Metadata for a group of binary slack bits added for one inequality.

    Attributes
    ----------
    constraint_name : str
        Name of the original constraint.
    sense : str
        ``'LE'`` or ``'GE'``.
    start_index : int
        First index in the expanded variable vector.
    num_bits : int
        Number of binary slack bits.
    max_slack : int
        Upper bound on the integer slack value.
    """

    constraint_name: str
    sense: str
    start_index: int
    num_bits: int
    max_slack: int


@dataclass
class CPAffineModel:
    """Compact lifted CP model used by FWAL.

    Attributes
    ----------
    C : np.ndarray
        Symmetric cost matrix (p x p) in the lifted formulation.
    constraint_matrices : list[np.ndarray]
        Symmetric matrices M_k such that A(W)_k = <M_k, W>.
    rhs : np.ndarray
        Right-hand side vector *v*.
    n_original : int
        Number of binary variables in the original problem.
    n_expanded : int
        Total binary variables (original + slack bits).
    p : int
        Lifted dimension  p = n_expanded + 1.
    slack_groups : list[SlackInfo]
        Metadata for each group of slack bits.
    """

    C: np.ndarray
    constraint_matrices: List[np.ndarray]
    rhs: np.ndarray
    n_original: int
    n_expanded: int
    p: int
    slack_groups: List[SlackInfo] = field(default_factory=list)


# ------------------------------------------------------------------ #
#   Helpers
# ------------------------------------------------------------------ #


def _symmetrise(M: np.ndarray) -> np.ndarray:
    return (M + M.T) / 2.0


def _compute_slack_bits(
    a: np.ndarray, b: float, sense
) -> Tuple[int, int]:
    """Return ``(max_slack, num_bits)`` for an inequality constraint.

    For LE (a^T x <= b): slack s = b - a^T x, max when a^T x is minimised.
    For GE (a^T x >= b): slack s = a^T x - b, max when a^T x is maximised.
    """
    if sense == LinearConstraint.Sense.LE:
        min_ax = sum(float(ai) for ai in a if ai < 0)
        max_slack = b - min_ax
    elif sense == LinearConstraint.Sense.GE:
        max_ax = sum(float(ai) for ai in a if ai > 0)
        max_slack = max_ax - b
    else:
        return 0, 0

    max_slack = max(int(math.ceil(max_slack)), 0)
    if max_slack == 0:
        return 0, 0
    num_bits = int(math.ceil(math.log2(max_slack + 1)))
    return max_slack, num_bits


# ------------------------------------------------------------------ #
#   Model builder
# ------------------------------------------------------------------ #


def build_cp_affine_model(problem: AbstractProblem) -> CPAffineModel:
    """Build the compact CP model ``(C, {M_k}, v)`` used by FWAL.

    Supports equality (==), less-or-equal (<=), and greater-or-equal (>=)
    constraints.  Inequality constraints are converted to equalities via
    binary slack variables as described in the Q-FWAL notes.
    """
    qp = problem.quadratic_binary_problem

    Q = _symmetrise(qp.objective.quadratic.to_array())
    c = qp.objective.linear.to_array()
    n = len(c)

    # ---- Determine slack variables ------------------------------------
    slack_groups: List[SlackInfo] = []
    total_slack_bits = 0

    for constraint in qp.linear_constraints:
        if constraint.sense == LinearConstraint.Sense.EQ:
            continue

        a_tmp = np.zeros(n, dtype=float)
        for idx, coeff in constraint.linear.to_dict().items():
            a_tmp[idx] = float(coeff)
        b_tmp = float(constraint.rhs)

        max_slack, num_bits = _compute_slack_bits(a_tmp, b_tmp, constraint.sense)

        if num_bits > 0:
            slack_groups.append(
                SlackInfo(
                    constraint_name=constraint.name,
                    sense=(
                        "LE"
                        if constraint.sense == LinearConstraint.Sense.LE
                        else "GE"
                    ),
                    start_index=n + total_slack_bits,
                    num_bits=num_bits,
                    max_slack=max_slack,
                )
            )
            total_slack_bits += num_bits

    n_expanded = n + total_slack_bits
    p = n_expanded + 1

    if total_slack_bits:
        logger.info(
            f"Inequality constraints expanded: "
            f"n={n} -> n_expanded={n_expanded} (p={p}), "
            f"{len(slack_groups)} slack groups, {total_slack_bits} bits"
        )

    # ---- Build cost matrix C ------------------------------------------
    # Paper: C = [[0, s^T], [s, Q]] with s = c/2, zero-padded for slack.
    s = 0.5 * c
    C = np.zeros((p, p), dtype=float)
    C[0, 1 : n + 1] = s
    C[1 : n + 1, 0] = s
    C[1 : n + 1, 1 : n + 1] = Q
    C = _symmetrise(C)

    # ---- Build constraint matrices (all symmetric) --------------------
    mats: List[np.ndarray] = []
    rhs_vals: List[float] = []
    slack_idx = 0

    for constraint in qp.linear_constraints:
        a_orig = np.zeros(n, dtype=float)
        for idx, coeff in constraint.linear.to_dict().items():
            a_orig[idx] = float(coeff)
        b = float(constraint.rhs)

        # Build expanded coefficient vector
        a_expanded = np.zeros(n_expanded, dtype=float)
        a_expanded[:n] = a_orig

        if constraint.sense != LinearConstraint.Sense.EQ:
            sg = slack_groups[slack_idx]
            slack_idx += 1
            for j in range(sg.num_bits):
                power = float(2**j)
                if sg.sense == "LE":
                    a_expanded[sg.start_index + j] = power
                else:  # GE
                    a_expanded[sg.start_index + j] = -power

        # 1) a_expanded^T x = b  (symmetric encoding in lifted space)
        M_eq_x = np.zeros((p, p), dtype=float)
        M_eq_x[0, 1:] = a_expanded / 2.0
        M_eq_x[1:, 0] = a_expanded / 2.0
        mats.append(M_eq_x)
        rhs_vals.append(b)

        # 2) Tr(A_expanded * X) = b^2   where A_expanded = a a^T
        M_eq_xx = np.zeros((p, p), dtype=float)
        M_eq_xx[1:, 1:] = np.outer(a_expanded, a_expanded)
        mats.append(_symmetrise(M_eq_xx))
        rhs_vals.append(b * b)

    # 3) diag(X) = x   for every expanded variable (symmetric encoding)
    for i in range(n_expanded):
        M_diag = np.zeros((p, p), dtype=float)
        M_diag[i + 1, i + 1] = 1.0
        M_diag[i + 1, 0] = -0.5
        M_diag[0, i + 1] = -0.5
        mats.append(M_diag)
        rhs_vals.append(0.0)

    # 4) W[0, 0] = 1
    M_hom = np.zeros((p, p), dtype=float)
    M_hom[0, 0] = 1.0
    mats.append(M_hom)
    rhs_vals.append(1.0)

    return CPAffineModel(
        C=C,
        constraint_matrices=mats,
        rhs=np.asarray(rhs_vals, dtype=float),
        n_original=n,
        n_expanded=n_expanded,
        p=p,
        slack_groups=slack_groups,
    )


# ------------------------------------------------------------------ #
#   Linear-algebra helpers
# ------------------------------------------------------------------ #


def apply_affine_map(
    constraint_matrices: List[np.ndarray],
    W: np.ndarray,
) -> np.ndarray:
    """Compute A(W) where A(W)_k = <M_k, W>."""
    return np.asarray(
        [float(np.sum(M * W)) for M in constraint_matrices], dtype=float
    )


def apply_affine_adjoint(
    constraint_matrices: List[np.ndarray],
    y: np.ndarray,
) -> np.ndarray:
    """Compute A*(y) = sum_k y_k M_k."""
    adj = np.zeros_like(constraint_matrices[0], dtype=float)
    for coeff, M in zip(y, constraint_matrices):
        adj += float(coeff) * M
    return adj


def compute_fwal_gradient(
    C: np.ndarray,
    constraint_matrices: List[np.ndarray],
    rhs: np.ndarray,
    W: np.ndarray,
    y: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Compute FWAL primal gradient G_t (symmetric).

    G_t = C + A*(y_t) + beta_t * A*(A(W_t) - v)
    """
    residual = apply_affine_map(constraint_matrices, W) - rhs
    grad = C + apply_affine_adjoint(constraint_matrices, y)
    grad += beta * apply_affine_adjoint(constraint_matrices, residual)
    return _symmetrise(grad)


def evaluate_cp_objective(C: np.ndarray, W: np.ndarray) -> float:
    """Evaluate Tr(C W)."""
    return float(np.sum(C * W))
