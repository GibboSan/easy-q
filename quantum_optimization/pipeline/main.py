import logging
import time

import numpy as np
from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime import EstimatorV2, SamplerV2

from pipeline.backends import get_aer_from_backend, get_real_backend
from pipeline.plotter import Plotter
from pipeline.problems.abstract_problem import AbstractProblem
from pipeline.runtime import parameter_optimization, sample_circuit
from pipeline.utils import find_most_promising_feasible_bitstring, class_importer

logger = logging.getLogger("pipeline_logger")


def single_run(parameter_dict: dict) -> dict:

    seed = parameter_dict['seed']
    output_folder = parameter_dict['output_folder']
    backend_name = parameter_dict['backend_name']
    is_backend_fake = parameter_dict['is_backend_fake']
    problem_class = parameter_dict['problem_class']
    circuit_class = parameter_dict['circuit_class']
    num_layers = parameter_dict['num_layers']
    num_starting_points = parameter_dict['num_starting_points']
    lower_bound = parameter_dict['lower_bound']
    upper_bound = parameter_dict['upper_bound']
    optimization_params = parameter_dict['optimization_params']
    use_cache = parameter_dict['use_cache']
    cache_filename = parameter_dict['cache_filename']
    cache_save_every = parameter_dict['cache_save_every']
    num_estimator_shots = parameter_dict['num_estimator_shots']
    num_sampler_shots = parameter_dict['num_sampler_shots']
    problem_params = parameter_dict['problem_params']

    logger.info(f"Using seed {seed}")
    np.random.seed(seed)
    algorithm_globals.random_seed = seed

    logger.info(f"Output will be written in {output_folder}")
    plotter = Plotter(f"{output_folder}/plots")

    backend = get_aer_from_backend(seed)
    if backend_name:
        logger.info(f"Building backend {backend_name}")
        backend = get_real_backend(backend_name)
        if is_backend_fake:
            backend = get_aer_from_backend(seed, backend)

    ProblemClass = class_importer("pipeline.problems", problem_class)
    CircuitClass = class_importer("pipeline.qaoa_circuits", circuit_class)

    logger.info(f"Building problem {problem_class}")
    problem: AbstractProblem = ProblemClass(seed, problem_params)

    num_qubits = problem.hamiltonian.num_qubits
    logger.info(f"The problem has {num_qubits} logic qubits")

    qaoa = CircuitClass(seed, problem, num_qubits, num_layers, backend)

    logger.info(f"Building QAOA circuit {circuit_class} with {num_layers} layers")
    tic = time.perf_counter()
    qaoa.get_parameterized_circuit()
    circuit_creation_time = time.perf_counter() - tic

    logger.info(f"Transpiling QAOA circuit {circuit_class} for {backend_name}")
    tic = time.perf_counter()
    tqc = qaoa.transpile()
    circuit_transpilation_time = time.perf_counter() - tic

    used_qubits = set()
    for instr, qargs, _ in tqc.data:
        for qubit in qargs:
            used_qubits.add(qubit)

    physical_qubits = len(used_qubits)
    logger.info(f"The problem has {physical_qubits} physical qubits")

    logger.info(f"Optimizing gammas and betas with {num_starting_points} starting points within {lower_bound} and {upper_bound}")
    tic = time.perf_counter()
    optimal_params, objective_fun_vals_all, final_params_all = parameter_optimization(
        num_layers,
        num_starting_points,
        (lower_bound, upper_bound),
        optimization_params,
        EstimatorV2,
        num_estimator_shots,
        backend,
        tqc,
        qaoa.hamiltonian,
        use_cache,
        cache_filename,
        cache_save_every
    )
    circuit_optimization_time = time.perf_counter() - tic

    logger.info(f"The optimal parameter configuration is:")
    gammas = optimal_params[num_layers:]
    betas = optimal_params[:num_layers]
    logger.info(f"Gammas: {gammas}")
    logger.info(f"Betas: {betas}")

    tic = time.perf_counter()
    final_qc = qaoa.get_bound_circuit(gammas, betas)
    circuit_bounding_time = time.perf_counter() - tic

    logger.info(f"Sampling with {num_sampler_shots} shots")
    tic = time.perf_counter()
    final_distribution_bin = sample_circuit(final_qc, backend, SamplerV2, num_sampler_shots)
    circuit_sampling_time = time.perf_counter() - tic

    quantum_best = find_most_promising_feasible_bitstring(final_distribution_bin, problem)

    classic_best = problem.get_best_solution()

    logger.info(f"Classic optimal solution: {classic_best}")
    if quantum_best:
        logger.info(f"Quantum best solution: {quantum_best} with frequency {final_distribution_bin[quantum_best[0]]}")
    else:
        logger.info(f"QAOA has failed, no feasible solution has been sampled")

    most_frequent = list(final_distribution_bin.items())[0]

    most_frequent_bitstring = most_frequent[0]
    most_frequent_objective = problem.evaluate_cost(most_frequent_bitstring)
    most_frequent_frequency = most_frequent[1]

    logger.info(f"Most frequent bitstring is: ('{most_frequent_bitstring}', {most_frequent_objective}) with frequency {most_frequent_frequency}")

    logger.info(f"Classic walltime: {problem.wall_time} [{problem.status}]")
    logger.info(f"{circuit_creation_time = }")
    logger.info(f"{circuit_optimization_time = }")
    logger.info(f"{circuit_bounding_time = }")
    logger.info(f"{circuit_sampling_time = }")

    quantum_walltime = sum([circuit_creation_time, circuit_optimization_time, circuit_bounding_time, circuit_sampling_time, circuit_transpilation_time])

    logger.info(f"Quantum walltime: {quantum_walltime}")
    logger.info("Generating plots...")
    plotter.draw_circuit(final_qc, "circuit.png")
    plotter.plot_parameter_optimization(objective_fun_vals_all, final_params_all, "parameters_optimization.png")
    plotter.plot_bitstring_distribution(final_distribution_bin, problem, "bitstring_histogram.png")
    plotter.generate_frequency_report(final_distribution_bin, problem, "freq_report.csv")
    logger.info("Terminated.")

    return {
        "seed": seed,
        "problem_class": problem_class,
        "circuit_class": circuit_class,
        "backend": backend_name,
        "logic_qubits": problem.hamiltonian.num_qubits,
        "physical_qubits": physical_qubits,
        "layers": num_layers,
        "transpiled_circuit_depth": tqc.depth(),
        "starting_points": num_starting_points,
        "optimal_parameters": list(optimal_params),
        "best_classic_bistring": classic_best[0],
        "best_classic_objective": classic_best[1],
        "best_classic_status": problem.status,
        "best_classic_walltime": problem.wall_time,
        "best_quantum_bitstring": quantum_best[0] if quantum_best else None,
        "best_quantum_objective": quantum_best[1] if quantum_best else None,
        "best_quantum_frequency": final_distribution_bin[quantum_best[0]] if quantum_best else None,
        "most_frequent_bitstring": most_frequent_bitstring,
        "most_frequent_objective": most_frequent_objective,
        "most_frequent_frequency": most_frequent_frequency,
        "circuit_creation_time": circuit_creation_time,
        "circuit_transpilation_time": circuit_transpilation_time,
        "circuit_optimization_time": circuit_optimization_time,
        "circuit_bounding_time": circuit_bounding_time,
        "circuit_sampling_time": circuit_sampling_time,
        "total_quantum_walltime": quantum_walltime
    }
