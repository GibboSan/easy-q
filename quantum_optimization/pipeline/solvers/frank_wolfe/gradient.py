"""
Gradient computation for Quadratic Binary Optimisation.

Given a QBO problem:

    min  x^T Q x  +  c^T x
    s.t. Ax = b,  x in {0,1}^n

The gradient of the continuous relaxation is:

    nabla f(x) = (Q + Q^T) x + c

When Q is symmetric this simplifies to  2 Q x + c.
"""

from typing import Tuple

import numpy as np

from pipeline.problems.abstract_problem import AbstractProblem


def build_Q_and_c(problem: AbstractProblem) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract the Q matrix and c vector from an AbstractProblem's
    quadratic binary formulation (*before* QUBO penalty encoding).

    The objective is:  f(x) = x^T Q x + c^T x + offset

    Parameters
    ----------
    problem : AbstractProblem
        Problem instance whose ``quadratic_binary_problem`` attribute
        carries the objective coefficients and linear constraints.

    Returns
    -------
    Q : np.ndarray, shape (n, n)
        Symmetric quadratic coefficient matrix.
    c : np.ndarray, shape (n,)
        Linear coefficient vector.
    offset : float
        Constant term in the objective.
    """
    qp = problem.quadratic_binary_problem

    Q = qp.objective.quadratic.to_array()
    c = qp.objective.linear.to_array()
    offset = qp.objective.constant

    # Symmetrise (Qiskit may store only one triangle)
    Q = (Q + Q.T) / 2.0

    return Q, c, float(offset)


def compute_gradient(Q: np.ndarray, c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of  f(x) = x^T Q x + c^T x .

        nabla f(x) = (Q + Q^T) x + c

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        Symmetric quadratic coefficient matrix.
    c : np.ndarray, shape (n,)
        Linear coefficient vector.
    x : np.ndarray, shape (n,)
        Current continuous point in [0, 1]^n.

    Returns
    -------
    grad : np.ndarray, shape (n,)
    """
    # Q is already symmetric, so (Q + Q^T) = 2Q
    return (Q + Q.T) @ x + c
