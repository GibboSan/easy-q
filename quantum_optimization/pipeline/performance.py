import logging
import time
import os

import numpy as np
from qiskit_algorithms.utils import algorithm_globals
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel
from qiskit.transpiler import PassManager

# Compatibility shim for qiskit_ibm_runtime expecting calc_final_ops in qiskit
from qiskit.transpiler.passes.utils import remove_final_measurements as _rfm
if not hasattr(_rfm, "calc_final_ops"):
    def calc_final_ops(dag, final_op_names):
        final_ops = _rfm.RemoveFinalMeasurements()._calc_final_ops(dag)
        names = set(final_op_names)
        return [node for node in final_ops if getattr(node, "name", None) in names]

    _rfm.calc_final_ops = calc_final_ops

from qiskit_ibm_runtime.transpiler.passes import ConvertISAToClifford
from qiskit_ibm_runtime.debug_tools import Neat

from pipeline.backends import get_aer_from_backend, get_real_backend, build_pruned_noise_model
from pipeline.problems.abstract_problem import AbstractProblem
from pipeline.utils import ( 
    class_importer, 
    get_circuit_metrics
)
from pipeline.mneat import MNeat

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

    seed: int = parameter_dict['seed']
    output_folder: str = parameter_dict['output_folder']
    backend_name: str = parameter_dict['backend_name']
    is_backend_fake: bool = parameter_dict['is_backend_fake']
    problem_class: str = parameter_dict['problem_class']
    solver_class: str = parameter_dict.get('solver_class', 'None')
    if solver_class.lower() != 'qaoasolver':
        raise ValueError(f"Expected solver_class='QAOASolver' in params_with_solver.yaml, got {solver_class!r}")
    solver_params = parameter_dict.get('solver_params', {})
    circuit_class: str = solver_params.get('circuit_class', 'QAOACircuit')
    num_layers: int = solver_params.get('num_layers', 1)
    num_estimator_shots: int = solver_params.get('num_estimator_shots', 10000)
    problem_params: dict = parameter_dict['problem_params']
    performance_params: dict = parameter_dict.get('performance', {})
    only_clifford: bool = performance_params.get('only_clifford', parameter_dict.get('only_clifford', True))
    pruned_noise: bool = performance_params.get('pruned_noise', parameter_dict.get('pruned_noise', False))
    use_general_observables: dict = performance_params.get('use_general_observables', parameter_dict.get('use_general_observables', False))
    mitigation_params: dict = performance_params.get('mitigation_params', parameter_dict.get('mitigation_params', {}))

    mitigation = True
    if not mitigation_params:
        mitigation = False

    logger.info(f"Using seed {seed}")
    np.random.seed(seed)
    algorithm_globals.random_seed = seed

    logger.info(f"Output will be written in {output_folder}")

    ###
    # Backend
    ###
    backend = get_aer_from_backend(seed)
    if backend_name:
        logger.info(f"Building backend {backend_name} {'(AerSimulator)' if is_backend_fake else '(real qpu)'}")
        backend = get_real_backend(backend_name)
        if is_backend_fake:
            backend = get_aer_from_backend(seed, backend)
    
    ideal_backend = get_aer_from_backend(seed, backend, noise_model=NoiseModel())

    ###
    # Problem and Circuit
    ###
    ProblemClass = class_importer("pipeline.problems", problem_class)
    CircuitClass = class_importer("pipeline.qaoa_circuits", circuit_class)

    logger.info(f"Building {problem_class}")
    problem: AbstractProblem = ProblemClass(seed, problem_params)

    num_qubits = problem.hamiltonian.num_qubits
    logger.info(f"The problem has {num_qubits} logic qubits")

    qaoa = CircuitClass(seed, problem, num_qubits, num_layers, backend)

    logger.info(f"Building {circuit_class} with {num_layers} layers")
    tic = time.perf_counter()
    qc = qaoa.get_parameterized_circuit()
    circuit_creation_time = time.perf_counter() - tic

    logger.info("Getting virtual circuit metrics")
    qc_metrics = get_circuit_metrics(qc, speedup=True)

    logger.info(f"Transpiling {circuit_class} for {backend_name}")
    tic = time.perf_counter()
    tqc = transpile_circuit(qc, backend, seed, optimization_level=3)
    transpilation_time = time.perf_counter() - tic

    logger.info("Getting transpiled circuit metrics")
    tqc_metrics = get_circuit_metrics(tqc, speedup=True)

    logger.info("Assigning random parameters to the transpiled circuit")
    rng = np.random.default_rng(seed)
    dummy_params = rng.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2], size=tqc.num_parameters)
    assigned_tqc = tqc.assign_parameters(dummy_params)

    logger.info("Preparing Hamiltonian for Estimator")
    isa_hamiltonian = qaoa.hamiltonian.apply_layout(tqc.layout)
    pub = [(assigned_tqc, isa_hamiltonian)]

    if use_general_observables:
        logger.info(f"Preparing general observables for Estimator")
        observables = ['I' * i + 'Z' + 'I' * (qc.num_qubits - i - 1) for i in range(qc.num_qubits)]
        isa_observables = [SparsePauliOp(o).apply_layout(tqc.layout) for o in observables]
        pub_observables = [(assigned_tqc, isa_observables)]

    ###
    # Optional backend with pruned noise model
    ###
    if pruned_noise:
        logger.info("Building pruned noise model")
        pruned_noise_model = build_pruned_noise_model(backend, assigned_tqc)
        backend = get_aer_from_backend(seed, backend, noise_model=pruned_noise_model)

    ###
    # Run Estimator on ideal and noisy backend for original circuit
    ###
    if not only_clifford:

        logger.info(f"Estimator run on {backend_name} without noise (ideal)")
        tic = time.perf_counter()
        ideal_value = estimator_run(pub, ideal_backend)
        ideal_time = time.perf_counter() - tic

        logger.info(f"Estimator run on {backend_name} with full noise")
        tic = time.perf_counter()
        noisy_value = estimator_run(pub, backend, num_estimator_shots)
        noisy_time = time.perf_counter() - tic

    ###
    # Run MNeat on ideal and noisy backend for Clifford circuit
    ###
    logger.info(f"Clifford Noise Analysis with MNEAT")
    mneat = MNeat(backend=backend)

    tic = time.perf_counter()
    clifford_pub = mneat.to_clifford(pub)
    # clifford_tqc = transpile_circuit(clifford_pub[0].circuit, backend, seed)
    # clifford_pub = [(clifford_tqc, clifford_pub[0].observables)]
    # clifford_pub = mneat.to_clifford(clifford_pub)
    clifford_transpilation_time = time.perf_counter() - tic
    clifford_tqc_metrics = get_circuit_metrics(clifford_pub[0].circuit, speedup=True)

    logger.info(f"Virtual QC Metrics:    {qc_metrics}")
    logger.info(f"Transpiled QC Metrics: {tqc_metrics}")
    logger.info(f"Clifford QC Metrics:   {clifford_tqc_metrics}")

    logger.info(f"Hamiltonian estimation with MNEAT on Clifford circuit")

    logger.info(f"Ideal simulation run")
    tic = time.perf_counter()
    clifford_ideal_value = float(mneat.ideal_sim(clifford_pub, cliffordize=False, seed_simulator=seed)[0].vals.item())
    clifford_ideal_time = time.perf_counter() - tic

    logger.info(f"Noisy simulation run")
    tic = time.perf_counter()
    clifford_noisy_value = float(mneat.noisy_sim(clifford_pub, cliffordize=False, seed_simulator=seed)[0].vals.item())
    clifford_noisy_time = time.perf_counter() - tic

    error_pct_noisy = 100 * abs(clifford_noisy_value - clifford_ideal_value) / abs(clifford_ideal_value) if clifford_ideal_value != 0 else None

    ###
    # Mitigation
    ###
    if mitigation:
        technique = mitigation_params.get('technique', 'zne')
        technique_params = mitigation_params.get('technique_params', {})
        logger.info(f"Mitigation simulation run with technique: {technique}")
        kwargs = {}
        
        # Handle ZNE-specific parameters
        if technique == 'zne':
            from mitiq.zne.inference import LinearFactory, RichardsonFactory, PolyFactory, ExpFactory, PolyExpFactory, AdaExpFactory
            
            factory_type = technique_params.get('factory', 'linear')
            noise_values = technique_params.get('noise_values', [1,2,3])
            
            logger.info(f"ZNE factory: {factory_type}, noise_values: {noise_values}")
            
            # Create factory based on type
            if factory_type.lower() == 'richardson':
                factory = RichardsonFactory(scale_factors=noise_values)
            elif factory_type.lower() == 'poly':
                factory = PolyFactory(scale_factors=noise_values)
            elif factory_type.lower() == 'exp':
                factory = ExpFactory(scale_factors=noise_values)
            elif factory_type.lower() == 'polyexp':
                factory = PolyExpFactory(scale_factors=noise_values)
            elif factory_type.lower() == 'adaexp':
                factory = AdaExpFactory(scale_factors=noise_values)
            else:  # linear (default)
                factory = LinearFactory(scale_factors=noise_values)
            
            kwargs['factory'] = factory
        
        tic = time.perf_counter()
        clifford_mitigated_value = float(mneat.mitigated_sim(clifford_pub, cliffordize=False, seed_simulator=seed, technique=technique, kwargs=kwargs)[0].vals.item())
        clifford_mitigated_time = time.perf_counter() - tic

        error_pct_mitigated = 100 * abs(clifford_mitigated_value - clifford_ideal_value) / abs(clifford_ideal_value) if clifford_ideal_value != 0 else None
        
        # Save factory plot if ZNE and plot can be generated
        if technique == 'zne' and 'factory' in kwargs:
            try:
                import matplotlib.pyplot as plt
                os.makedirs(output_folder, exist_ok=True)
                factory = kwargs['factory']
                
                # Plot the factory (this creates a figure with the extrapolation curve)
                factory.plot_fit()
                plot_path = os.path.join(output_folder, 'zne_factory_fit.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved ZNE factory plot to {plot_path}")
            except Exception as e:
                logger.warning(f"Could not save ZNE factory plot: {e}")

    ###
    # Print results
    ###
    if not only_clifford:
        logger.info(f"Ideal Estimator Value:           {ideal_value}")
        logger.info(f"Noisy Estimator Value:      {noisy_value}")
    logger.info(f"Clifford Ideal Estimator Value:       {clifford_ideal_value}")
    logger.info(f"Clifford Noisy Estimator Value:       {clifford_noisy_value}")
    logger.info(f"Clifford Mitigated Estimator Value:   {clifford_mitigated_value if mitigation else 'N/A'}")

    logger.info(f"Error % of Noisy vs Ideal:       {error_pct_noisy:.2f}%" if error_pct_noisy is not None else "N/A")
    logger.info(f"Error % of Mitigated vs Ideal:   {error_pct_mitigated:.2f}%" if error_pct_mitigated is not None else "N/A")

    logger.info(f"Circuit creation time:       {circuit_creation_time:.4f} seconds")
    logger.info(f"Transpilation time:          {transpilation_time:.4f} seconds")
    logger.info(f"Clifford Transpilation time: {clifford_transpilation_time:.4f} seconds")

    if not only_clifford:
        logger.info(f"Ideal Estimation time:           {ideal_time:.4f} seconds")
        logger.info(f"Noisy Estimation time:      {noisy_time:.4f} seconds")
    logger.info(f"Clifford Ideal Estimation time:       {clifford_ideal_time:.4f} seconds")
    logger.info(f"Clifford Noisy Estimation time:       {clifford_noisy_time:.4f} seconds")
    logger.info(f"Clifford Mitigated Estimation time:   {clifford_mitigated_time:.4f} seconds" if mitigation else "N/A")

    logger.info("ESTIMATOR PERFORMANCE RUN COMPLETED.")

    return {
        "seed": seed,
        "problem_class": problem_class,
        "problem_params": problem_params,
        "circuit_class": circuit_class,
        "layers": num_layers,
        "backend": backend_name,
        "logic_qubits": problem.hamiltonian.num_qubits,
        "backend_qubits": backend.num_qubits,
        "virtual_qc_metrics": qc_metrics,
        "transpiled_qc_metrics": tqc_metrics,
        "clifford_qc_metrics": clifford_tqc_metrics,
        "performance": {
            "only_clifford": only_clifford,
            "pruned_noise": pruned_noise,
        },
        "mitigation_technique": technique if mitigation else None,
        "mitigation_params": technique_params if mitigation else None,
        "num_estimator_shots": num_estimator_shots,
        "ideal_energy": ideal_value if not only_clifford else None,
        "noisy_energy": noisy_value if not only_clifford else None,
        "clifford_ideal_energy": clifford_ideal_value,
        "clifford_noisy_energy": clifford_noisy_value,
        "clifford_mitigated_energy": clifford_mitigated_value if mitigation else None,
        "error_pct_noisy": error_pct_noisy,
        "error_pct_mitigated": error_pct_mitigated,
        "circuit_creation_time": circuit_creation_time,
        "transpilation_time": transpilation_time,
        "transpilation_clifford_time": clifford_transpilation_time,
        "ideal_estimation_time": ideal_time if not only_clifford else None,
        "noisy_estimation_time": noisy_time if not only_clifford else None,
        "clifford_ideal_estimation_time": clifford_ideal_time,
        "clifford_noisy_estimation_time": clifford_noisy_time,
        "clifford_mitigated_estimation_time": clifford_mitigated_time if mitigation else None,
        "zne_factory_plot": os.path.join(output_folder, 'zne_factory_fit.png') if (mitigation and technique == 'zne') else None
    }
