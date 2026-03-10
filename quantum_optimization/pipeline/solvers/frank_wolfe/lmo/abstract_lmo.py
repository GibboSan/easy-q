"""Abstract interface for Linear Minimisation Oracles (LMO) in FWAL.

Every LMO implementation solves the unconstrained QUBO

    min_{w in {0,1}^p}  w^T G w

and returns the rank-1 matrix H = w w^T (an extreme point of Delta^p).
"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class AbstractLMO(ABC):
    """Base class for LMO backends."""

    @abstractmethod
    def solve(self, G: np.ndarray, seed: int) -> Dict:
        """Solve the LMO for the given gradient matrix G.

        Parameters
        ----------
        G : np.ndarray
            Symmetric gradient matrix of shape (p, p).
        seed : int
            Reproducibility seed for this call.

        Returns
        -------
        dict
            ``w``              – np.ndarray, optimal binary vector
            ``bitstring``      – str, binary representation of w
            ``vertex_matrix``  – np.ndarray, H = w w^T
            ``lmo_objective``  – float, w^T G w
            ``lmo_time``       – float, wall-clock seconds
            ``solve_mode``     – str, identifier for the solver used
        """
        pass
