"""Linear minimisation oracle for FWAL in lifted space.

The oracle solves (approximately):

    min_{w in {0,1}^p} w^T G w

and returns H = w w^T, which is an extreme point of Delta^p.
"""

import time
from typing import Dict

import numpy as np


def _objective_from_bits(G: np.ndarray, w: np.ndarray) -> float:
    return float(w @ G @ w)


def _bits_to_bitstring(w: np.ndarray) -> str:
    return "".join(str(int(v)) for v in w)


def _solve_exact_bruteforce(G: np.ndarray) -> tuple[np.ndarray, float]:
    p = G.shape[0]
    best_w = np.zeros(p, dtype=float)
    best_val = float("inf")
    for mask in range(1 << p):
        w = np.array([(mask >> i) & 1 for i in range(p)], dtype=float)
        val = _objective_from_bits(G, w)
        if val < best_val:
            best_val = val
            best_w = w
    return best_w, best_val


def _solve_random_local_search(
    G: np.ndarray,
    rng: np.random.Generator,
    num_random_samples: int,
    local_search_steps: int,
) -> tuple[np.ndarray, float]:
    p = G.shape[0]
    best_w = rng.integers(0, 2, size=p).astype(float)
    best_val = _objective_from_bits(G, best_w)

    for _ in range(max(1, num_random_samples)):
        w = rng.integers(0, 2, size=p).astype(float)
        val = _objective_from_bits(G, w)

        improved = True
        steps = 0
        while improved and steps < local_search_steps:
            improved = False
            steps += 1
            for i in range(p):
                w_try = w.copy()
                w_try[i] = 1.0 - w_try[i]
                val_try = _objective_from_bits(G, w_try)
                if val_try + 1e-12 < val:
                    w = w_try
                    val = val_try
                    improved = True

        if val < best_val:
            best_w = w
            best_val = val

    return best_w, best_val


def fwal_lmo(
    G: np.ndarray,
    seed: int,
    *,
    max_exact_bits: int = 22,
    num_random_samples: int = 4096,
    local_search_steps: int = 200,
) -> Dict:
    """Solve the FWAL LMO and return H = w w^T.

    Parameters
    ----------
    G : np.ndarray
        Current FWAL gradient matrix.
    seed : int
        Random seed for heuristic search.
    max_exact_bits : int
        Use brute-force if p <= this threshold.
    num_random_samples : int
        Number of random restarts in heuristic mode.
    local_search_steps : int
        Max local-improvement passes in heuristic mode.
    """
    tic = time.perf_counter()

    G = (G + G.T) / 2.0
    p = G.shape[0]

    if p <= max_exact_bits:
        w, lmo_value = _solve_exact_bruteforce(G)
        solve_mode = "exact"
    else:
        rng = np.random.default_rng(seed)
        w, lmo_value = _solve_random_local_search(
            G,
            rng,
            num_random_samples=num_random_samples,
            local_search_steps=local_search_steps,
        )
        solve_mode = "heuristic"

    H = np.outer(w, w)

    return {
        "w": w,
        "bitstring": _bits_to_bitstring(w),
        "vertex_matrix": H,
        "lmo_objective": float(lmo_value),
        "lmo_time": time.perf_counter() - tic,
        "solve_mode": solve_mode,
    }
