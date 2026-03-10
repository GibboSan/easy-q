"""QAOA-based LMO for solving the FWAL linear minimisation sub-problem.

Uses a Quantum Approximate Optimization Algorithm (QAOA) circuit to
approximately solve the unconstrained QUBO  min w^T G w.
Requires a quantum backend (real or Aer simulator) and follows the same
parameter-optimisation / sampling pipeline used in the main QAOA solver.
"""

import logging
import time
from typing import Dict, Optional

import numpy as np
from qiskit.providers import Backend
from qiskit_optimization import QuadraticProgram

from pipeline.solvers.frank_wolfe.lmo.abstract_lmo import AbstractLMO

logger = logging.getLogger("pipeline_logger")


def _qubo_matrix_to_hamiltonian(G: np.ndarray):
    """Convert a symmetric QUBO matrix G to an Ising SparsePauliOp.

    Builds a ``QuadraticProgram`` with diagonal terms as linear coefficients
    and off-diagonal terms as quadratic coefficients, then calls
    ``to_ising()`` to obtain the Hamiltonian.
    """
    G_sym = (G + G.T) / 2.0
    p = G_sym.shape[0]

    qp = QuadraticProgram("lmo_qubo")
    for i in range(p):
        qp.binary_var(f"w_{i}")

    # For binary variables x_i^2 = x_i, so separate diagonal (linear)
    # from off-diagonal (quadratic, upper triangle only with factor 2).
    linear = {f"w_{i}": float(G_sym[i, i]) for i in range(p)}
    quadratic = {}
    for i in range(p):
        for j in range(i + 1, p):
            val = 2.0 * float(G_sym[i, j])
            if abs(val) > 1e-15:
                quadratic[(f"w_{i}", f"w_{j}")] = val

    qp.minimize(linear=linear, quadratic=quadratic)

    hamiltonian, offset = qp.to_ising()
    return hamiltonian, offset


class _HamiltonianWrapper:
    """Minimal adapter to supply a Hamiltonian to QAOACircuit."""

    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian


class QAOALMO(AbstractLMO):
    """LMO that solves  min w^T G w  via QAOA.

    Parameters
    ----------
    backend : Backend
        Qiskit backend (real or simulator).
    circuit_class : type
        QAOACircuit subclass to build the ansatz.
    num_layers : int
        Number of QAOA layers.
    num_starting_points : int
        Number of random parameter starting points.
    lower_bound : float
        Lower bound for initial parameter values.
    upper_bound : float
        Upper bound for initial parameter values.
    optimization_params : dict
        Optimizer configuration (``optimizer``, ``tolerance``, etc.).
    num_estimator_shots : int
        Shots for the Estimator.
    num_sampler_shots : int
        Shots for the Sampler.
    """

    def __init__(
        self,
        backend: Backend,
        circuit_class: type,
        num_layers: int = 1,
        num_starting_points: int = 3,
        lower_bound: float = 0.0,
        upper_bound: float = 6.2832,
        optimization_params: Optional[dict] = None,
        num_estimator_shots: int = 1024,
        num_sampler_shots: int = 4096,
    ):
        self.backend = backend
        self.circuit_class = circuit_class
        self.num_layers = num_layers
        self.num_starting_points = num_starting_points
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.optimization_params = optimization_params or {
            "optimizer": "COBYLA",
            "tolerance": 1e-3,
        }
        self.num_estimator_shots = num_estimator_shots
        self.num_sampler_shots = num_sampler_shots

    def solve(self, G: np.ndarray, seed: int) -> Dict:
        # Deferred imports to avoid heavy load when this LMO is not used
        from qiskit_ibm_runtime import EstimatorV2, SamplerV2
        from pipeline.runtime import parameter_optimization, sample_circuit

        tic = time.perf_counter()
        G_sym = (G + G.T) / 2.0
        p = G_sym.shape[0]

        # 1. Convert QUBO matrix → Ising Hamiltonian
        hamiltonian, offset = _qubo_matrix_to_hamiltonian(G_sym)
        wrapper = _HamiltonianWrapper(hamiltonian)

        # 2. Build QAOA circuit
        qaoa = self.circuit_class(
            seed=seed,
            problem=wrapper,
            num_qubits=hamiltonian.num_qubits,
            num_layers=self.num_layers,
            backend=self.backend,
        )
        qaoa.get_parameterized_circuit()
        tqc = qaoa.transpile()

        # 3. Parameter optimisation
        logger.info(
            f"QAOALMO: optimising {hamiltonian.num_qubits}-qubit QAOA "
            f"with {self.num_layers} layers"
        )
        optimal_params, optimal_energy, _, _, _ = parameter_optimization(
            n_layer=self.num_layers,
            n_starting_point=self.num_starting_points,
            bounds=(self.lower_bound, self.upper_bound),
            optimization_params=self.optimization_params,
            Estimator=EstimatorV2,
            estimator_shots=self.num_estimator_shots,
            backend=self.backend,
            circuit=tqc,
            hamiltonian=qaoa.hamiltonian,
            use_cache=False,
            cache_filename="",
        )

        # 4. Sample the bound circuit
        gammas = optimal_params[self.num_layers:]
        betas = optimal_params[:self.num_layers]
        final_qc = qaoa.get_bound_circuit(gammas, betas)

        distribution = sample_circuit(
            final_qc, self.backend, SamplerV2, self.num_sampler_shots
        )

        # 5. Pick the bitstring with the lowest w^T G w
        best_val = float("inf")
        best_bs = "0" * p
        for bs in distribution:
            w_cand = np.array([int(b) for b in bs], dtype=float)
            if len(w_cand) != p:
                continue
            val = float(w_cand @ G_sym @ w_cand)
            if val < best_val:
                best_val = val
                best_bs = bs

        w = np.array([int(b) for b in best_bs], dtype=float)
        H = np.outer(w, w)

        logger.info(
            f"QAOALMO: best objective = {best_val:.6f}, "
            f"time = {time.perf_counter() - tic:.2f}s"
        )

        return {
            "w": w,
            "bitstring": best_bs,
            "vertex_matrix": H,
            "lmo_objective": float(best_val),
            "lmo_time": time.perf_counter() - tic,
            "solve_mode": "qaoa",
        }
