"""Heuristic LMO using random restarts with greedy local search.

Generates random binary vectors and improves each via single-bit flips.
Practical for problems too large for brute-force enumeration.
"""

import time
from typing import Dict

import numpy as np

from pipeline.solvers.frank_wolfe.lmo.abstract_lmo import AbstractLMO


class LocalSearchLMO(AbstractLMO):
    """Heuristic LMO via random restarts with greedy 1-flip local search.

    Parameters
    ----------
    num_random_samples : int
        Number of random starting points.
    local_search_steps : int
        Maximum number of full-pass improvement sweeps per starting point.
    """

    def __init__(
        self,
        num_random_samples: int = 4096,
        local_search_steps: int = 200,
    ):
        self.num_random_samples = num_random_samples
        self.local_search_steps = local_search_steps

    def solve(self, G: np.ndarray, seed: int) -> Dict:
        tic = time.perf_counter()
        G = (G + G.T) / 2.0
        p = G.shape[0]
        rng = np.random.default_rng(seed)

        best_w = rng.integers(0, 2, size=p).astype(float)
        best_val = float(best_w @ G @ best_w)

        for _ in range(max(1, self.num_random_samples)):
            w = rng.integers(0, 2, size=p).astype(float)
            val = float(w @ G @ w)

            improved = True
            steps = 0
            while improved and steps < self.local_search_steps:
                improved = False
                steps += 1
                for i in range(p):
                    w_try = w.copy()
                    w_try[i] = 1.0 - w_try[i]
                    val_try = float(w_try @ G @ w_try)
                    if val_try + 1e-12 < val:
                        w = w_try
                        val = val_try
                        improved = True

            if val < best_val:
                best_w = w
                best_val = val

        H = np.outer(best_w, best_w)
        return {
            "w": best_w,
            "bitstring": "".join(str(int(v)) for v in best_w),
            "vertex_matrix": H,
            "lmo_objective": float(best_val),
            "lmo_time": time.perf_counter() - tic,
            "solve_mode": "local_search",
        }
