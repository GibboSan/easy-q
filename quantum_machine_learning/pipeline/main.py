import sys
import logging

from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime import SamplerV2 as SamplerRT
from qiskit.primitives import StatevectorSampler as StateSampler
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeMumbaiV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from pipeline.methods import fit_model_parallel
from pipeline.classic import run_classic_SVC

logger = logging.getLogger("pipeline_logger")

def single_run(parameters_dict: str, dataset: dict) -> None:

    method = parameters_dict['method_type']
    logger.info(f"Methods used {method}")

    seed = parameters_dict['seed']
    algorithm_globals.random_seed = seed
    logger.info(f"Global seed is set to {seed}")

    backend = parameters_dict['backend']

    ##################################
    #### Quantum
    ##################################

    if backend == 'Aer':
        sampler = SamplerRT(mode=AerSimulator(seed_simulator=seed))
        pass_manager = None
    elif backend == 'StateVector':
        sampler = StateSampler()
        pass_manager = None
    elif backend == 'AerMumbai':
        device_backend = AerSimulator.from_backend(FakeMumbaiV2(), seed_simulator=seed)
        sampler = SamplerRT(mode=device_backend)
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=device_backend)
    else:
        logger.error(f"Error: backend '{backend}' does not exist.")
        sys.exit(1)
        
    logger.info(f"{backend} Sampler initialized...")

    model_dict = fit_model_parallel(sampler, pass_manager, parameters_dict, **dataset)


    logger.info(f"Saving quantum '{method}' results")
    # Quantum Circuit details
    if "feature_map" in model_dict:
        feat_map_dict = {
                        "n_qubit_feat_map": len(model_dict['feature_map'].qubits),
                        "circuit_depth_feat_map": model_dict['feature_map'].depth(),
                        "circuit_n_gate_feat_map": sum(model_dict['feature_map'].count_ops().values()),
                        "circuit_n_cnot_feat_map": model_dict['feature_map'].count_ops().get("cx", 0)
                        }
    else:
        feat_map_dict = {
                        "n_qubit_feat_map": None,
                        "circuit_depth_feat_map": None,
                        "circuit_n_gate_feat_map": None,
                        "circuit_n_cnot_feat_map": None
                        }
        
    if "ansatz" in model_dict:
        ansatz_dict = {
                        "n_qubit_ansatz": len(model_dict['ansatz'].qubits),
                        "circuit_depth_ansatz": model_dict['ansatz'].depth(),
                        "circuit_n_gate_ansatz": sum(model_dict['ansatz'].count_ops().values()),
                        "circuit_n_cnot_ansatz": model_dict['ansatz'].count_ops().get("cx", 0)
                        }
    else:
        ansatz_dict = {
                        "n_qubit_ansatz": None,
                        "circuit_depth_ansatz": None,
                        "circuit_n_gate_ansatz": None,
                        "circuit_n_cnot_ansatz": None
                        }
        
    if "kernel" in model_dict:
        kernel_dict = { "walltime_kernel_train": model_dict['kernel']['elapsed_time_kernel_train'],
                        "walltime_kernel_test": model_dict['kernel']['elapsed_time_kernel_test']
                        }
        
        trained_model = {"model": model_dict["fit_obj"],
                         "kernel": {
                                    "kernel_train": model_dict['kernel']["kernel_train"],
                                    "kernel_test": model_dict['kernel']["kernel_test"]
                                    }
                        }

    else:
        kernel_dict = { "walltime_kernel_train": None,
                        "walltime_kernel_test": None
                        }
        
        trained_model = {"model": model_dict["fit_obj"],
                         "objective_func": model_dict["objective_func"]}


    # Fit Results
    scores = {"train_score": model_dict['train_score'],
              "test_score": model_dict['test_score'],
              "walltime": model_dict['elapsed_time']
              }

    quantum_results = {"quantum_results": {**scores, **kernel_dict}
                        }    

    ##################################
    #### Classical SVC
    ##################################

    logger.info(f"Running classical SVC")

    train_score_c, test_score_c, elapsed_time_c = run_classic_SVC(**dataset)

    classic_results = {
            "classic_SVC": {
                "train_score": train_score_c,
                "test_score": test_score_c,
                "walltime": elapsed_time_c
            }
        }

    logger.info(f"Saving classical SVC results")

    # Save the results
    output_dict = {**parameters_dict, **feat_map_dict, **ansatz_dict, **quantum_results, **classic_results}


    return output_dict, trained_model

