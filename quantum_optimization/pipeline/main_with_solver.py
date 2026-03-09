import logging
import time

import numpy as np
from qiskit_algorithms.utils import algorithm_globals

from pipeline.backends import get_aer_from_backend, get_real_backend
from pipeline.plotter import Plotter
from pipeline.problems.abstract_problem import AbstractProblem
from pipeline.solvers.abstract_solver import AbstractSolver
from pipeline.utils import class_importer

logger = logging.getLogger("pipeline_logger")


def single_run_with_solver(parameter_dict: dict) -> dict:

    seed = parameter_dict["seed"]
    output_folder = parameter_dict["output_folder"]
    backend_name = parameter_dict["backend_name"]
    is_backend_fake = parameter_dict["is_backend_fake"]

    problem_class = parameter_dict["problem_class"]
    problem_params = parameter_dict["problem_params"]

    solver_class = parameter_dict["solver_class"]
    solver_params = parameter_dict.get("solver_params", {})

    # QFWSolver keeps circuit_class in its signature for compatibility.
    circuit_class = parameter_dict.get("circuit_class", "QAOACircuit")

    logger.info(f"Using seed {seed}")
    np.random.seed(seed)
    algorithm_globals.random_seed = seed

    logger.info(f"Output will be written in {output_folder}")
    plotter = Plotter(f"{output_folder}/plots")

    backend = get_aer_from_backend(seed)
    if backend_name:
        logger.info(
            f"Building backend {backend_name} "
            f"{'(AerSimulator)' if is_backend_fake else '(real qpu)'}"
        )
        backend = get_real_backend(backend_name)
        if is_backend_fake:
            backend = get_aer_from_backend(seed, backend)

    ProblemClass = class_importer("pipeline.problems", problem_class)
    CircuitClass = class_importer("pipeline.qaoa_circuits", circuit_class)
    SolverClass = class_importer("pipeline.solvers", solver_class, compute_classfile_name=False)

    logger.info(f"Building problem {problem_class}")
    problem: AbstractProblem = ProblemClass(seed, problem_params)

    num_qubits = problem.hamiltonian.num_qubits
    logger.info(f"The problem has {num_qubits} logic qubits")

    logger.info(f"Building solver {solver_class}")
    solver: AbstractSolver = SolverClass(
        problem=problem,
        backend=backend,
        seed=seed,
        circuit_class=CircuitClass,
        **solver_params,
    )

    logger.info("Running solver")
    tic = time.perf_counter()
    solver_output = solver.solve()
    solver_walltime = time.perf_counter() - tic

    best_bitstring = solver_output.get("best_bitstring")
    best_objective = solver_output.get("best_objective")

    classic_best = problem.get_best_solution()

    logger.info(f"Classic optimal solution: {classic_best}")
    logger.info(f"Solver best solution: ({best_bitstring}, {best_objective})")
    logger.info(f"Classic walltime: {problem.wall_time} [{problem.status}]")
    logger.info(f"{solver_walltime = }")

    # Plot convergence as one optimization run.
    convergence = solver_output.get("convergence", {})
    cp_history = convergence.get("cp_objective_history")
    if cp_history:
        plotter.plot_parameter_optimization(
            objective_fun_vals_all=[cp_history],
            final_params_all=[[float(len(cp_history))]],
            filename="solver_convergence.png",
        )

    # Plot/report a degenerate final distribution with the solver-best bitstring.
    if best_bitstring:
        final_distribution_bin = {best_bitstring: 1.0}
        plotter.plot_bitstring_distribution(final_distribution_bin, problem, "bitstring_histogram.png")
        plotter.generate_frequency_report(final_distribution_bin, problem, "freq_report.csv")

    logger.info("Terminated.")

    return {
        "seed": seed,
        "problem_class": problem_class,
        "solver_class": solver_class,
        "circuit_class": circuit_class,
        "backend": backend_name,
        "logic_qubits": num_qubits,
        "best_classic_bistring": classic_best[0],
        "best_classic_objective": classic_best[1],
        "best_classic_status": problem.status,
        "best_classic_walltime": problem.wall_time,
        "best_solver_bitstring": best_bitstring,
        "best_solver_objective": best_objective,
        "solver_walltime": solver_walltime,
        "solver_output": solver_output,
    }
