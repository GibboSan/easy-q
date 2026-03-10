"""LMO factory and dispatcher for FWAL.

Provides :func:`create_lmo` to build an LMO backend from a configuration
dictionary, and :func:`fwal_lmo` as a backward-compatible one-shot
interface that creates and invokes the chosen LMO.

Available methods:

- ``"auto"``          – brute-force if p <= *max_exact_bits*, else local search
- ``"bruteforce"``    – exact enumeration (O(2^p), small problems only)
- ``"local_search"``  – random restarts + 1-flip local improvement
- ``"cplex"``         – IBM CPLEX via docplex
- ``"qaoa"``          – QAOA on a quantum backend
"""

import logging
from typing import Dict, Optional

import numpy as np
from qiskit.providers import Backend

from pipeline.solvers.frank_wolfe.lmo import (
    AbstractLMO,
    BruteforceLMO,
    LocalSearchLMO,
    CplexLMO,
    QAOALMO,
)

logger = logging.getLogger("pipeline_logger")


# ------------------------------------------------------------------ #
#   Factory
# ------------------------------------------------------------------ #


def create_lmo(
    method: str = "auto",
    *,
    backend: Optional[Backend] = None,
    circuit_class: Optional[type] = None,
    # bruteforce / auto threshold
    max_exact_bits: int = 22,
    # local search
    num_random_samples: int = 4096,
    local_search_steps: int = 200,
    # cplex
    cplex_time_limit: float = 60.0,
    # qaoa
    qaoa_num_layers: int = 1,
    qaoa_num_starting_points: int = 3,
    qaoa_lower_bound: float = 0.0,
    qaoa_upper_bound: float = 6.2832,
    qaoa_optimization_params: Optional[dict] = None,
    qaoa_num_estimator_shots: int = 1024,
    qaoa_num_sampler_shots: int = 4096,
    **_extra,
) -> AbstractLMO:
    """Build an LMO backend from keyword configuration.

    Parameters
    ----------
    method : str
        ``"auto"`` (default), ``"bruteforce"``, ``"local_search"``,
        ``"cplex"``, or ``"qaoa"``.
    """
    method = method.lower()

    if method == "bruteforce":
        return BruteforceLMO()

    if method == "local_search":
        return LocalSearchLMO(num_random_samples, local_search_steps)

    if method == "cplex":
        return CplexLMO(time_limit=cplex_time_limit)

    if method == "qaoa":
        if backend is None:
            raise ValueError("QAOA LMO requires a quantum backend.")
        if circuit_class is None:
            raise ValueError("QAOA LMO requires a circuit_class.")
        return QAOALMO(
            backend=backend,
            circuit_class=circuit_class,
            num_layers=qaoa_num_layers,
            num_starting_points=qaoa_num_starting_points,
            lower_bound=qaoa_lower_bound,
            upper_bound=qaoa_upper_bound,
            optimization_params=qaoa_optimization_params,
            num_estimator_shots=qaoa_num_estimator_shots,
            num_sampler_shots=qaoa_num_sampler_shots,
        )

    if method == "auto":
        return _AutoLMO(
            max_exact_bits=max_exact_bits,
            num_random_samples=num_random_samples,
            local_search_steps=local_search_steps,
        )

    raise ValueError(f"Unknown LMO method: '{method}'")


# ------------------------------------------------------------------ #
#   Auto selector
# ------------------------------------------------------------------ #


class _AutoLMO(AbstractLMO):
    """Auto-selects brute-force or local-search based on problem dimension."""

    def __init__(self, max_exact_bits, num_random_samples, local_search_steps):
        self._bf = BruteforceLMO()
        self._ls = LocalSearchLMO(num_random_samples, local_search_steps)
        self._threshold = max_exact_bits

    def solve(self, G: np.ndarray, seed: int) -> Dict:
        p = G.shape[0]
        if p <= self._threshold:
            return self._bf.solve(G, seed)
        return self._ls.solve(G, seed)


# ------------------------------------------------------------------ #
#   Backward-compatible one-shot interface
# ------------------------------------------------------------------ #


def fwal_lmo(G: np.ndarray, seed: int, **lmo_params) -> Dict:
    """Create an LMO from *lmo_params* and invoke it for *G*.

    This function preserves the original call signature for scripts that
    have not yet been migrated to the class-based LMO API.
    """
    lmo = create_lmo(**lmo_params)
    return lmo.solve(G, seed)
