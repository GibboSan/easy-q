"""QAOA-based solver for constrained binary quadratic problems.

Wraps the full QAOA pipeline (circuit creation, transpilation, parameter
optimisation, sampling, distribution analysis) into an ``AbstractSolver``
interface.  The logic mirrors ``pipeline.main.single_run`` without
modifying it.

This solver can be used standalone to solve any ``AbstractProblem``.
"""

import logging
import time
import os
from typing import Any, Dict, Optional

import numpy as np
from qiskit.providers import Backend
from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime import EstimatorV2, SamplerV2

from pipeline.problems.abstract_problem import AbstractProblem

from pipeline.runtime import parameter_optimization, sample_circuit
from pipeline.solvers.abstract_solver import AbstractSolver
from pipeline.utils import (
    class_importer,
    analyze_distribution,
    compute_approximation_ratio,
    get_circuit_metrics,
)
from pipeline.plotter import Plotter

logger = logging.getLogger("pipeline_logger")


class QAOASolver(AbstractSolver):
    """Solve a constrained QBO via QAOA.

    Parameters
    ----------
    seed : int
        Random seed.
    output_folder: str
        Folder to write solver outputs (e.g. convergence logs, plots).
    backend : Backend
        Quantum backend (real or Aer simulator).
    circuit_class : type
        QAOACircuit subclass to build the ansatz.
    num_layers : int
        Number of QAOA layers.
    num_starting_points : int
        Number of random parameter starting points.
    lower_bound : float
        Lower bound for initial parameter search.
    upper_bound : float
        Upper bound for initial parameter search.
    optimization_params : dict
        Optimizer configuration forwarded to ``parameter_optimization``.
    num_estimator_shots : int
        Shots for the Estimator primitive.
    num_sampler_shots : int
        Shots for the Sampler primitive.
    use_cache : bool
        Whether to use the optimisation cache.
    cache_filename : str
        Path for the cache YAML file.
    cache_save_every : int
        Save cache every N evaluations.
    """

    def __init__(
        self,
        seed: int,
        backend: Backend,
        circuit_class: str,
        num_layers: int = 1,
        num_starting_points: int = 5,
        lower_bound: float = 0.0,
        upper_bound: float = 6.2832,
        optimization_params: Optional[dict] = None,
        num_estimator_shots: int = 1024,
        num_sampler_shots: int = 4096,
        use_cache: bool = False,
        cache_filename: str = "qaoa_cache.yaml",
        cache_save_every: int = 1,
        output_folder: str = "",
        **kwargs,
    ):
        logger.info(f"Initializing QAOASolver with seed {seed}")
        super().__init__(seed=seed, output_folder=output_folder)
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
        self.use_cache = use_cache
        self.cache_filename = cache_filename
        self.cache_save_every = cache_save_every
    # ------------------------------------------------------------------ #

    def solve(self, problem: AbstractProblem) -> Dict[str, Any]:
        np.random.seed(self.seed)
        algorithm_globals.random_seed = self.seed

        num_qubits = problem.hamiltonian.num_qubits

        tic_total = time.perf_counter()

        # ---- Circuit creation -------------------------------------------
        logger.info(
            f"QAOASolver: building QAOA circuit {self.circuit_class} with {self.num_layers} layers"
        )
        CircuitClass = class_importer(
            "pipeline.qaoa_circuits", 
            self.circuit_class
        )
        qaoa = CircuitClass(
            self.seed,
            problem,
            num_qubits,
            self.num_layers,
            self.backend,
        )
        tic = time.perf_counter()
        qc = qaoa.get_parameterized_circuit()
        circuit_creation_time = time.perf_counter() - tic
        qc_metrics = get_circuit_metrics(qc)

        # ---- Transpilation ----------------------------------------------
        logger.info(
            "QAOASolver: transpiling circuit"
        )
        tic = time.perf_counter()
        tqc = qaoa.transpile()
        circuit_transpilation_time = time.perf_counter() - tic
        tqc_metrics = get_circuit_metrics(tqc)
        logger.info(
            f"QAOASolver: the problem has {tqc_metrics['num_active_qubits']} physical qubits"
        )

        # ---- Parameter optimisation -------------------------------------
        logger.info(
            f"QAOASolver: optimising with bounds "
            f"[{self.lower_bound}, {self.upper_bound}] "
            f"and {self.num_starting_points} starting points"
        )
        tic = time.perf_counter()
        (
            optimal_params,
            optimal_energy,
            optimal_nfev,
            objective_fun_vals_all,
            final_params_all,
        ) = parameter_optimization(
            n_layer=self.num_layers,
            n_starting_point=self.num_starting_points,
            bounds=(self.lower_bound, self.upper_bound),
            optimization_params=self.optimization_params,
            Estimator=EstimatorV2,
            estimator_shots=self.num_estimator_shots,
            backend=self.backend,
            circuit=tqc,
            hamiltonian=qaoa.hamiltonian,
            use_cache=self.use_cache,
            cache_filename=self.cache_filename,
            cache_save_every=self.cache_save_every,
        )
        circuit_optimization_time = time.perf_counter() - tic

        gammas = optimal_params[self.num_layers:]
        betas = optimal_params[:self.num_layers]

        # ---- Sampling ---------------------------------------------------
        tic = time.perf_counter()
        final_qc = qaoa.get_bound_circuit(gammas, betas)
        circuit_bounding_time = time.perf_counter() - tic

        logger.info(f"QAOASolver: sampling with {self.num_sampler_shots} shots")
        tic = time.perf_counter()
        final_distribution_bin = sample_circuit(
            final_qc, self.backend, SamplerV2, self.num_sampler_shots
        )
        circuit_sampling_time = time.perf_counter() - tic

        total_time = time.perf_counter() - tic_total

        # ---- Analysis ---------------------------------------------------
        classic_best = problem.get_best_solution()
        (
            quantum_best, 
            most_frequent, 
            avg_energy, 
            success_probability
        ) = analyze_distribution(
            final_distribution_bin,
            problem,
            optimal_cost=classic_best[1],
        )

        approx_ratio = compute_approximation_ratio(
            avg_energy, classic_best[1]
        )

        quantum_walltime = sum([
            circuit_creation_time,
            circuit_transpilation_time,
            circuit_optimization_time,
            circuit_bounding_time,
            circuit_sampling_time,
        ])

        logger.info(
            f"QAOASolver: /n "
            f"Classic optimal solution: {classic_best}\n"
            f"QAOA best solution: {quantum_best}\n"
            f"Most frequent solution: ('{most_frequent[0]}', {most_frequent[1]}) with frequency {most_frequent[2]}\n"
            f"Classic walltime: {problem.wall_time:.2f}s [{problem.status}]\n"
            f"Quantum walltime: {quantum_walltime:.2f}s\n"
            f"QAOA walltime: {total_time:.2f}s"
        )

        # ---- Plotting ------------------------------------------------
        if self.output_folder:
            output_folder = os.path.join(self.output_folder, "plots")
            logger.info(f"QAOASolver: saving plots to {output_folder}")
            plotter = Plotter(f"{output_folder}")

            plotter.draw_circuit(final_qc, "circuit.png")
            plotter.plot_parameter_optimization(
                objective_fun_vals_all,
                final_params_all,
                "parameters_optimization.png",
            )
            plotter.plot_bitstring_distribution(
                final_distribution_bin,
                problem,
                "bitstring_histogram.png",
            )
            plotter.generate_frequency_report(
                final_distribution_bin,
                problem,
                "freq_report.csv",
            )

        #---- Return results ------------------------------------------------
        logger.info("QAOASolver: terminated.")

        return {
            "virtual_qubits": qc_metrics["num_active_qubits"],
            "physical_qubits": tqc_metrics["num_active_qubits"],
            "classic_best_bitstring": classic_best[0],
            "classic_best_objective": classic_best[1],
            "classic_best_status": problem.status,
            "solver_best_bitstring": quantum_best[0],
            "solver_best_objective": quantum_best[1] if quantum_best[0] else None,
            "solver_best_frequency": quantum_best[2],
            "most_frequent_bitstring": most_frequent[0],
            "most_frequent_objective": most_frequent[1],
            "most_frequent_frequency": most_frequent[2],
            "average_energy": avg_energy,
            "approximation_ratio": approx_ratio,
            "effective_success_probability": success_probability,
            "optimal_parameters": list(optimal_params),
            "optimal_estimator_energy": float(optimal_energy),
            "optimization_nfev": int(optimal_nfev),
            "virtual_depth": qc.depth(),
            "transpiled_depth": tqc.depth(),
            "circuit_creation_time": circuit_creation_time,
            "circuit_transpilation_time": circuit_transpilation_time,
            "circuit_optimization_time": circuit_optimization_time,
            "circuit_bounding_time": circuit_bounding_time,
            "circuit_sampling_time": circuit_sampling_time,
            "quantum_walltime": quantum_walltime,
            "solver_walltime": total_time,
            "classic_walltime": problem.wall_time,
        }
