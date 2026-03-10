"""Brute-force exact LMO for small problem instances.

Enumerates all 2^p binary vectors to find the global minimum of w^T G w.
Only practical when p is small (typically p <= 22).
"""

import time
from typing import Dict

import numpy as np

from pipeline.solvers.frank_wolfe.lmo.abstract_lmo import AbstractLMO


class BruteforceLMO(AbstractLMO):
    """Exact LMO via exhaustive enumeration over {0,1}^p."""

    def solve(self, G: np.ndarray, seed: int) -> Dict:
        tic = time.perf_counter()
        G = (G + G.T) / 2.0
        p = G.shape[0]

        best_w = np.zeros(p, dtype=float)
        best_val = float("inf")

        for mask in range(1 << p):
            w = np.array([(mask >> i) & 1 for i in range(p)], dtype=float)
            val = float(w @ G @ w)
            if val < best_val:
                best_val = val
                best_w = w

        H = np.outer(best_w, best_w)
        return {
            "w": best_w,
            "bitstring": "".join(str(int(v)) for v in best_w),
            "vertex_matrix": H,
            "lmo_objective": float(best_val),
            "lmo_time": time.perf_counter() - tic,
            "solve_mode": "bruteforce",
        }
