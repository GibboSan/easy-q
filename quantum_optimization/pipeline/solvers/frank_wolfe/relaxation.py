"""Initialization and lifted-variable extraction utilities for FWAL.

Provides functions to build the initial primal iterate W_0, the initial
dual iterate y_0, and to extract the original-space blocks x and X from
the lifted matrix W (accounting for slack variables introduced by
inequality constraint expansion).
"""

import numpy as np


def build_initial_primal_matrix(p: int, mode: str = "zeros") -> np.ndarray:
    """Build the initial lifted primal matrix W_0 in Delta^p.

    Parameters
    ----------
    p : int
        Lifted dimension.
    mode : str
        ``"zeros"`` for W_0 = 0 (paper default) or ``"homogeneous"``
        for W_0 = e_1 e_1^T.
    """
    if mode == "zeros":
        return np.zeros((p, p), dtype=float)
    if mode == "homogeneous":
        W0 = np.zeros((p, p), dtype=float)
        W0[0, 0] = 1.0
        return W0
    raise ValueError(f"Unknown initialisation mode: '{mode}'")


def build_initial_dual_vector(d: int) -> np.ndarray:
    """Build the initial dual iterate y_0 = 0."""
    return np.zeros(d, dtype=float)


def extract_x_and_X_from_W(
    W: np.ndarray, n_original: int
) -> tuple:
    """Extract original-space x and X from the lifted matrix W.

    Parameters
    ----------
    W : np.ndarray
        Lifted matrix of shape (p, p).
    n_original : int
        Number of original binary variables (excluding slack bits).

    Returns
    -------
    x : np.ndarray
        Continuous relaxation of the original variable vector (length n_original).
    X : np.ndarray
        Corresponding (n_original x n_original) block of W.
    """
    x = W[1 : n_original + 1, 0].copy()
    X = W[1 : n_original + 1, 1 : n_original + 1].copy()
    return x, X


def symmetrise(W: np.ndarray) -> np.ndarray:
    """Numerical symmetrisation helper."""
    return (W + W.T) / 2.0
