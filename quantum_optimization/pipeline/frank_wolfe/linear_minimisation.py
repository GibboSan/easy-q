"""
Linear Minimisation Oracle (LMO) for the Quantum Frank-Wolfe algorithm.

Replaces the Adiabatic Quantum Computing (AQC) step from the original
Q-FW paper with a **QAOA-based** approach, reusing the circuit
infrastructure already available in ``pipeline.qaoa_circuits``.

At each Frank-Wolfe iteration the LMO solves:

    min_{s in C}  grad^T s

where C is the set of *feasible binary vectors*.  The linear objective
is encoded as a diagonal Hamiltonian and handed to QAOA; the resulting
samples are filtered for feasibility and the best vertex is returned.
"""

import logging
import time
from typing import Dict, Optional, Type

import numpy as np
from qiskit.providers import Backend
from qiskit_ibm_runtime import EstimatorV2, SamplerV2

from pipeline.problems.abstract_problem import AbstractProblem
from pipeline.qaoa_circuits.qaoa_circuit import QAOACircuit
from pipeline.runtime import parameter_optimization, sample_circuit
from pipeline.frank_wolfe.relaxation import LinearSubproblem, gradient_to_hamiltonian

logger = logging.getLogger("pipeline_logger")


def qaoa_lmo(
    grad: np.ndarray,
    problem: AbstractProblem,
    circuit_class: Type[QAOACircuit],
    backend: Backend,
    seed: int,
    *,
    num_layers: int = 1,
    num_starting_points: int = 3,
    bounds: tuple = (-np.pi, np.pi),
    optimization_params: Optional[dict] = None,
    num_estimator_shots: int = 1024,
    num_sampler_shots: int = 4096,
    use_cache: bool = False,
    cache_filename: str = "lmo_cache.yaml",
    cache_save_every: int = 1,
) -> Dict:
    """
    Solve the Frank-Wolfe LMO via QAOA.

    Finds an approximate solution to::

        min_{s in C}  grad^T s

    where *C* is the set of feasible binary vectors defined by the
    constraints of ``problem``.

    Parameters
    ----------
    grad : np.ndarray, shape (n,)
        Gradient vector at the current FW iterate.
    problem : AbstractProblem
        Original problem (provides constraint info and feasibility checking).
    circuit_class : Type[QAOACircuit]
        QAOA circuit variant (``QAOACircuit``, ``AncillaQAOACircuit``, …).
    backend : Backend
        Quantum backend for circuit execution.
    seed : int
        Random seed.
    num_layers : int
        Number of QAOA layers for the LMO.
    num_starting_points : int
        Multi-start optimisation attempts for QAOA parameters.
    bounds : tuple
        ``(lower, upper)`` for QAOA parameter initialisation.
    optimization_params : dict, optional
        Optimizer settings forwarded to :func:`pipeline.runtime.parameter_optimization`.
    num_estimator_shots : int
        Shots for the Estimator during parameter optimisation.
    num_sampler_shots : int
        Shots for the final Sampler call.
    use_cache : bool
        Whether to cache QAOA parameter optimisation runs.
    cache_filename : str
        Path for the cache file.
    cache_save_every : int
        Cache persistence frequency.

    Returns
    -------
    dict
        ``vertex``        – np.ndarray (n,), binary FW vertex
        ``bitstring``     – str, same vertex as a bitstring
        ``lmo_objective`` – float, value of grad^T s
        ``distribution``  – dict, full sampling distribution
        ``qaoa_energy``   – float, optimal QAOA energy
        ``lmo_time``      – float, wall-clock seconds for this LMO call
    """
    if optimization_params is None:
        optimization_params = {"optimizer": "COBYLA", "tolerance": 1e-3}

    tic = time.perf_counter()

    n = len(grad)

    # 1. Build the linear Hamiltonian for this FW iteration
    lmo_hamiltonian = gradient_to_hamiltonian(grad)

    # 2. Wrap the problem so QAOA circuits see the linear Hamiltonian
    #    but still have access to constraint metadata
    sub_problem = LinearSubproblem(problem, lmo_hamiltonian)

    # 3. Build & transpile the QAOA circuit
    qaoa = circuit_class(seed, sub_problem, n, num_layers, backend)
    qaoa.get_parameterized_circuit()
    tqc = qaoa.transpile()

    # 4. Optimise QAOA parameters
    optimal_params, optimal_energy, _, _, _ = parameter_optimization(
        num_layers,
        num_starting_points,
        bounds,
        optimization_params,
        EstimatorV2,
        num_estimator_shots,
        backend,
        tqc,
        qaoa.hamiltonian,
        use_cache,
        cache_filename,
        cache_save_every,
    )

    # 5. Sample the optimised circuit
    gammas = optimal_params[num_layers:]
    betas = optimal_params[:num_layers]
    final_qc = qaoa.get_bound_circuit(gammas, betas)
    distribution = sample_circuit(final_qc, backend, SamplerV2, num_sampler_shots)

    # 6. Pick the best *feasible* bitstring that minimises grad^T s
    best_bitstring = None
    best_lmo_value = float("inf")

    for bitstring in distribution:
        if problem.is_feasible(bitstring)[0]:
            s = np.array([int(b) for b in bitstring], dtype=float)
            lmo_val = float(grad @ s)
            if lmo_val < best_lmo_value:
                best_lmo_value = lmo_val
                best_bitstring = bitstring

    # Fallback: no feasible sample → take the most frequent bitstring
    if best_bitstring is None:
        logger.warning("LMO: no feasible bitstring sampled; using most frequent")
        best_bitstring = max(distribution, key=distribution.get)
        s = np.array([int(b) for b in best_bitstring], dtype=float)
        best_lmo_value = float(grad @ s)

    vertex = np.array([int(b) for b in best_bitstring], dtype=float)

    lmo_time = time.perf_counter() - tic

    return {
        "vertex": vertex,
        "bitstring": best_bitstring,
        "lmo_objective": best_lmo_value,
        "distribution": distribution,
        "qaoa_energy": float(optimal_energy),
        "lmo_time": lmo_time,
    }
