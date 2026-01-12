import logging
import time

import numpy as np
from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime import EstimatorV2
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel
from qiskit.transpiler import PassManager
from qiskit_ibm_runtime.transpiler.passes import ConvertISAToClifford
from qiskit_ibm_runtime.debug_tools import Neat

from pipeline.backends import get_aer_from_backend, get_real_backend, build_pruned_noise_model
from pipeline.problems.abstract_problem import AbstractProblem
from pipeline.utils import ( 
    class_importer, 
    get_circuit_metrics
)

logger = logging.getLogger("pipeline_logger")

def transpile_circuit(qc, backend, seed=None, optimization_level=3):

    if backend is None:
        raise ValueError("No backend specified for transpilation.")
    
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed
    )

    return pm.run(qc)


def estimator_run(pub, backend, shots: int = None) -> float:

    if isinstance(backend, AerSimulator):
        backend_options = {}
        if hasattr(backend, 'options'):
            if hasattr(backend.options, 'noise_model'):
                backend_options['noise_model'] = backend.options.noise_model
            if hasattr(backend.options, 'method'):
                backend_options['method'] = backend.options.method
        
        estimator = AerEstimator(options={"backend_options": backend_options})
        
    precision = 1.0 / np.sqrt(shots) if shots is not None else 0.0
    res = estimator.run(pub, precision=precision).result()[0]
    exact_value = res.data.evs

    return float(exact_value.item())


def estimator_performance_run(parameter_dict: dict) -> dict:

    seed = parameter_dict['seed']
    output_folder = parameter_dict['output_folder']
    backend_name = parameter_dict['backend_name']
    is_backend_fake = parameter_dict['is_backend_fake']
    problem_class = parameter_dict['problem_class']
    circuit_class = parameter_dict['circuit_class']
    num_layers = parameter_dict['num_layers']
    num_estimator_shots = parameter_dict['num_estimator_shots']
    problem_params = parameter_dict['problem_params']
    only_clifford = parameter_dict['only_clifford']

    logger.info(f"Using seed {seed}")
    np.random.seed(seed)
    algorithm_globals.random_seed = seed

    logger.info(f"Output will be written in {output_folder}")

    backend = get_aer_from_backend(seed)
    if backend_name:
        logger.info(f"Building backend {backend_name} {'(AerSimulator)' if is_backend_fake else '(real qpu)'}")
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
    qc = qaoa.get_parameterized_circuit()
    circuit_creation_time = time.perf_counter() - tic

    qc_metrics = get_circuit_metrics(qc)

    logger.info(f"Transpiling QAOA circuit {circuit_class} for {backend_name}")
    tic = time.perf_counter()
    tqc = transpile_circuit(qc, backend, seed)
    transpilation_time = time.perf_counter() - tic
    tqc_metrics = get_circuit_metrics(tqc)

    rng = np.random.default_rng(seed)
    dummy_params = rng.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2], size=tqc.num_parameters)
    assigned_tqc = tqc.assign_parameters(dummy_params)

    isa_hamiltonian = qaoa.hamiltonian.apply_layout(tqc.layout)
    pub = [(assigned_tqc, isa_hamiltonian)]

    pruned_noise_model =  build_pruned_noise_model(backend, assigned_tqc)


    if not only_clifford:

        logger.info(f"Estimator run on {backend_name}")
        tic = time.perf_counter()
        noisy_value = estimator_run(pub, backend, num_estimator_shots)
        noisy_time = time.perf_counter() - tic

        logger.info(f"Estimator run on {backend_name} without noise (ideal)")
        ideal_backend = get_aer_from_backend(seed, backend, noise_model=NoiseModel())
        tic = time.perf_counter()
        ideal_value = estimator_run(pub, ideal_backend)
        ideal_time = time.perf_counter() - tic

        logger.info(f"Estimator run on {backend_name} with pruned noise")
        pruned_backend = get_aer_from_backend(seed, backend, noise_model=pruned_noise_model)
        tic = time.perf_counter()
        pruned_value = estimator_run(pub, pruned_backend, num_estimator_shots)
        pruned_time = time.perf_counter() - tic


    logger.info(f"Clifford Noise Analysis with NEAT")
    neat = Neat(backend=backend)
    tic = time.perf_counter()
    clifford_pub = neat.to_clifford(pub)
    optimized_clifford_tqc = transpile_circuit(clifford_pub[0].circuit, backend, seed)
    clifford_pub = [(optimized_clifford_tqc, clifford_pub[0].observables)]
    clifford_pub = neat.to_clifford(clifford_pub)
    clifford_transpilation_time = time.perf_counter() - tic
    clifford_tqc_metrics = get_circuit_metrics(optimized_clifford_tqc)

    logger.info(f"Ideal simulation run on Clifford circuit")
    tic = time.perf_counter()
    clifford_ideal_value = float(neat.ideal_sim(clifford_pub, cliffordize=False, seed_simulator=seed)[0].vals.item())
    clifford_ideal_time = time.perf_counter() - tic

    logger.info(f"Noisy simulation run on Clifford circuit")
    tic = time.perf_counter()
    clifford_noisy_value = float(neat.noisy_sim(clifford_pub, cliffordize=False, seed_simulator=seed)[0].vals.item())
    clifford_noisy_time = time.perf_counter() - tic

    logger.info(f"Pruned noise simulation run on Clifford circuit")
    pruned_neat = Neat(backend=backend, noise_model=pruned_noise_model)
    tic = time.perf_counter()
    clifford_pruned_value = float(pruned_neat.noisy_sim(clifford_pub, cliffordize=False, seed_simulator=seed)[0].vals.item())
    clifford_pruned_time = time.perf_counter() - tic


    # # Stabilizer method on Clifford circuit (instead of Neat)

    # clifford_pm = PassManager([ConvertISAToClifford()])
    # tic = time.perf_counter()
    # clifford_qc = clifford_pm.run(assigned_tqc)
    # clifford_transpilation_time = time.perf_counter() - tic
    # stabilizer_ideal_backend = get_aer_from_backend(seed, backend, noise_model=NoiseModel(), method="stabilizer")
    # tic = time.perf_counter()
    # stabilizer_qc = transpile_circuit(clifford_qc, stabilizer_ideal_backend, seed)
    # stabilizer_transpilation_time = time.perf_counter() - tic
    # stabilizer_tqc_metrics = get_circuit_metrics(stabilizer_qc)

    # logger.info(f"Estimator run on {backend_name} with stabilizer ideal method on Clifford circuit")
    # tic = time.perf_counter()
    # stabilizer_ideal_value = estimator_run(stabilizer_qc, isa_hamiltonian, stabilizer_ideal_backend)
    # stabilizer_ideal_estimation_time = time.perf_counter() - tic

    # logger.info(f"Estimator run on {backend_name} with stabilizer method on Clifford circuit with full noise")
    # noise_model = NoiseModel.from_backend(backend, thermal_relaxation=False)
    # stabilizer_noisy_backend = get_aer_from_backend(seed, backend, noise_model=noise_model, method="stabilizer")
    # tic = time.perf_counter()
    # stabilizer_noisy_value = estimator_run(stabilizer_qc, isa_hamiltonian, stabilizer_noisy_backend, num_estimator_shots)
    # stabilizer_noisy_estimation_time = time.perf_counter() - tic

    # logger.info(f"Estimator run on {backend_name} with stabilizer method on Clifford circuit with pruned noise")
    # pruned_stabilizer_backend = get_aer_from_backend(seed, backend, noise_model=pruned_noise_model, method="stabilizer")
    # tic = time.perf_counter()
    # stabilizer_pruned_value = estimator_run(stabilizer_qc, isa_hamiltonian, pruned_stabilizer_backend, num_estimator_shots)
    # stabilizer_pruned_estimation_time = time.perf_counter() - tic

    logger.info(f"Virtual QC Metrics:    {qc_metrics}")
    logger.info(f"Transpiled QC Metrics: {tqc_metrics}")
    logger.info(f"Clifford QC Metrics:   {clifford_tqc_metrics}")

    if not only_clifford:
        logger.info(f"Ideal Estimator Value:           {ideal_value}")
        logger.info(f"Reference Estimator Value:       {noisy_value}")
        logger.info(f"Pruned Noise Estimator Value:    {pruned_value}")
    logger.info(f"Clifford Ideal Estimator Value:  {clifford_ideal_value}")
    logger.info(f"Clifford Noisy Estimator Value:  {clifford_noisy_value}")
    logger.info(f"Clifford Pruned Estimator Value: {clifford_pruned_value}")

    logger.info(f"Circuit creation time:       {circuit_creation_time:.4f} seconds")
    logger.info(f"Transpilation time:          {transpilation_time:.4f} seconds")
    logger.info(f"Clifford Transpilation time: {clifford_transpilation_time:.4f} seconds")

    if not only_clifford:
        logger.info(f"Ideal Estimation time:           {ideal_time:.4f} seconds")
        logger.info(f"Noisy Estimation time:           {noisy_time:.4f} seconds")
        logger.info(f"Pruned Noise Estimation time:    {pruned_time:.4f} seconds")
    logger.info(f"Clifford Ideal Estimation time:  {clifford_ideal_time:.4f} seconds")
    logger.info(f"Clifford Noisy Estimation time:  {clifford_noisy_time:.4f} seconds")
    logger.info(f"Clifford Pruned Estimation time: {clifford_pruned_time:.4f} seconds")

    logger.info("ESTIMATOR PERFORMANCE RUN COMPLETED.")


    return {
        "seed": seed,
        "problem_class": problem_class,
        "circuit_class": circuit_class,
        "backend": backend_name,
        "logic_qubits": problem.hamiltonian.num_qubits,
        "backend_qubits": backend.num_qubits,
        "virtual_qc_metrics": qc_metrics,
        "transpiled_qc_metrics": tqc_metrics,
        "clifford_qc_metrics": clifford_tqc_metrics,
        "layers": num_layers,
        "num_estimator_shots": num_estimator_shots,
        "ideal_energy": ideal_value if not only_clifford else None,
        "noisy_energy": noisy_value if not only_clifford else None,
        "pruned_energy": pruned_value if not only_clifford else None,
        "clifford_ideal_energy": clifford_ideal_value,
        "clifford_noisy_energy": clifford_noisy_value,
        "clifford_pruned_energy": clifford_pruned_value,
        "circuit_creation_time": circuit_creation_time,
        "transpilation_time": transpilation_time,
        "transpilation_clifford_time": clifford_transpilation_time,
        "ideal_estimation_time": ideal_time if not only_clifford else None,
        "noisy_estimation_time": noisy_time if not only_clifford else None,
        "pruned_estimation_time": pruned_time if not only_clifford else None,
        "clifford_ideal_estimation_time": clifford_ideal_time,
        "clifford_noisy_estimation_time": clifford_noisy_time,
        "clifford_pruned_estimation_time": clifford_pruned_time
    }
