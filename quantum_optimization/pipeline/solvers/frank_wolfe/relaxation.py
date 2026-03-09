"""Initialization and lifted-variable extraction utilities for FWAL."""

import numpy as np


def build_initial_primal_matrix(p: int, mode: str = "zeros") -> np.ndarray:
    """Build the initial lifted primal matrix W_0 in Delta^p.

    Parameters
    ----------
    p : int
        Lifted dimension.
    mode : str
        ``"zeros"`` for W_0 = 0 (paper default in practice) or
        ``"homogeneous"`` for W_0 = e_1 e_1^T.
    """
    if mode == "zeros":
        return np.zeros((p, p), dtype=float)
    if mode == "homogeneous":
        W0 = np.zeros((p, p), dtype=float)
        W0[0, 0] = 1.0
        return W0
    raise ValueError(f"Unknown initialisation mode: '{mode}'")


def build_initial_dual_vector(d: int) -> np.ndarray:
    """Build the initial dual iterate y_0."""
    return np.zeros(d, dtype=float)


def extract_x_and_X_from_W(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract x and X blocks from the lifted matrix W.

    W is interpreted as:

        [ 1   x^T ]
        [ x    X  ]
    """
    x = W[1:, 0].copy()
    X = W[1:, 1:].copy()
    return x, X


def symmetrise(W: np.ndarray) -> np.ndarray:
    """Numerical symmetrisation helper."""
    return (W + W.T) / 2.0
