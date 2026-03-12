import logging
import time

import numpy as np
from qiskit_algorithms.utils import algorithm_globals

from pipeline.backends import get_aer_from_backend, get_real_backend
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

    logger.info(f"Using seed {seed}")
    np.random.seed(seed)
    algorithm_globals.random_seed = seed

    logger.info(f"Output will be written in {output_folder}")

    # ---- Backend -----------------------------------------------------
    backend = get_aer_from_backend(seed)
    if backend_name:
        logger.info(
            f"Building backend {backend_name} "
            f"{'(AerSimulator)' if is_backend_fake else '(real qpu)'}"
        )
        backend = get_real_backend(backend_name)
        if is_backend_fake:
            backend = get_aer_from_backend(seed, backend)

    # ---- Problem -----------------------------------------------------
    ProblemClass = class_importer("pipeline.problems", problem_class)
    logger.info(f"Building problem {problem_class}")
    problem: AbstractProblem = ProblemClass(seed, problem_params)

    num_qubits = problem.hamiltonian.num_qubits
    logger.info(f"The problem has {num_qubits} logic qubits")

    # ---- Solver ------------------------------------------------------
    logger.info(f"Building solver {solver_class}")
    SolverClass = class_importer(
        "pipeline.solvers", solver_class, compute_classfile_name=False,
    )
    params = dict(solver_params)
    
    solver: AbstractSolver = SolverClass(
        seed=seed,
        backend=backend,
        output_folder=output_folder, 
        **params
    )

    # ---- Run ---------------------------------------------------------
    logger.info("Running solver")
    solver_output = solver.solve(problem)

    logger.info("Terminated.")

    return {
        "seed": seed,
        "problem_class": problem_class,
        "solver_class": solver_class,
        "backend": backend_name,
        "logic_qubits": num_qubits,
        **solver_output,
    }
