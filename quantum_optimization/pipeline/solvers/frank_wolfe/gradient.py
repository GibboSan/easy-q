"""FWAL primitives in the lifted copositive space.

This module implements the Section 3--4 model ingredients from
Yurtsever et al. (2022): construction of the compact CP model

    min_W Tr(CW)  s.t. A(W) = v,  W in Delta^p

and the FWAL primal gradient

    G_t = C + A^*(y_t) + beta_t A^*(A(W_t) - v).
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from pipeline.problems.abstract_problem import AbstractProblem


@dataclass
class CPAffineModel:
    """Compact lifted model used by FWAL.

    Attributes
    ----------
    C : np.ndarray
        Cost matrix in the lifted formulation.
    constraint_matrices : list[np.ndarray]
        Linear operators M_k implementing A(W)_k = <M_k, W>.
    rhs : np.ndarray
        Right-hand side vector v.
    n : int
        Number of binary variables in the original QBO.
    p : int
        Lifted dimension p = n + 1.
    """

    C: np.ndarray
    constraint_matrices: List[np.ndarray]
    rhs: np.ndarray
    n: int
    p: int


def _symmetrise(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2.0


def build_cp_affine_model(problem: AbstractProblem) -> CPAffineModel:
    """Build the compact CP model (C, A, v) used by FWAL.

    Notes
    -----
    The model follows the paper's equality-constrained derivation.
    For now this implementation rejects non-equality constraints.
    """
    qp = problem.quadratic_binary_problem

    Q = _symmetrise(qp.objective.quadratic.to_array())
    c = qp.objective.linear.to_array()
    n = len(c)
    p = n + 1

    # The paper uses x^T Q x + 2 s^T x, while Qiskit exposes c^T x.
    # Hence s = c / 2.
    s = 0.5 * c

    C = np.zeros((p, p), dtype=float)
    C[0, 1:] = s
    C[1:, 0] = s
    C[1:, 1:] = Q
    C = _symmetrise(C)

    mats: List[np.ndarray] = []
    rhs: List[float] = []

    for constraint in qp.linear_constraints:
        if constraint.sense != constraint.Sense.EQ:
            raise NotImplementedError(
                "Faithful FWAL implementation currently supports equality constraints only."
            )

        a = np.zeros(n, dtype=float)
        for idx, coeff in constraint.linear.to_dict().items():
            a[idx] = float(coeff)
        b = float(constraint.rhs)

        # a^T x = b  ->  <M_eq_x, W> = b, where x = W[1:, 0]
        m_eq_x = np.zeros((p, p), dtype=float)
        m_eq_x[1:, 0] = a
        mats.append(m_eq_x)
        rhs.append(b)

        # Tr((a a^T) X) = b^2, where X = W[1:, 1:]
        m_eq_xx = np.zeros((p, p), dtype=float)
        m_eq_xx[1:, 1:] = np.outer(a, a)
        mats.append(m_eq_xx)
        rhs.append(b * b)

    # diag(X) = x
    for i in range(n):
        m_diag = np.zeros((p, p), dtype=float)
        m_diag[i + 1, i + 1] = 1.0
        m_diag[i + 1, 0] = -1.0
        mats.append(m_diag)
        rhs.append(0.0)

    # W_{1,1} = 1 (1-indexed in paper, [0,0] in code)
    m_hom = np.zeros((p, p), dtype=float)
    m_hom[0, 0] = 1.0
    mats.append(m_hom)
    rhs.append(1.0)

    return CPAffineModel(
        C=C,
        constraint_matrices=mats,
        rhs=np.asarray(rhs, dtype=float),
        n=n,
        p=p,
    )


def apply_affine_map(
    constraint_matrices: List[np.ndarray],
    W: np.ndarray,
) -> np.ndarray:
    """Compute A(W) where A(W)_k = <M_k, W>."""
    return np.asarray([float(np.sum(M * W)) for M in constraint_matrices], dtype=float)


def apply_affine_adjoint(
    constraint_matrices: List[np.ndarray],
    y: np.ndarray,
) -> np.ndarray:
    """Compute A^*(y) = sum_k y_k M_k."""
    if len(constraint_matrices) != len(y):
        raise ValueError("Size mismatch in apply_affine_adjoint")
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
    """Compute FWAL primal gradient matrix G_t."""
    residual = apply_affine_map(constraint_matrices, W) - rhs
    grad = C + apply_affine_adjoint(constraint_matrices, y)
    grad += beta * apply_affine_adjoint(constraint_matrices, residual)
    return _symmetrise(grad)


def evaluate_cp_objective(C: np.ndarray, W: np.ndarray) -> float:
    """Evaluate Tr(CW)."""
    return float(np.sum(C * W))
