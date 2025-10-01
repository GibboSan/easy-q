import time
import numpy as np

from sklearn.svm import SVC

from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.providers import Backend
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms.classifiers import VQC
import qiskit_machine_learning.optimizers as optimizers

from IPython.display import clear_output


import logging
logger = logging.getLogger("pipeline_logger")


CIRCUIT_TYPES = {
    "ZZFeatureMap": {
        "function": ZZFeatureMap
    },
    "RealAmplitudes": {
        "function": RealAmplitudes
    }
}

OPTIMIZERS = {
    "COBYLA" : optimizers.COBYLA,
    "SLSQP" : optimizers.SLSQP,
    "L_BFGS_B" : optimizers.L_BFGS_B,
}

def get_circuit(num_features: int, circuit_type: str, params: dict) -> QuantumCircuit:

    circuit = CIRCUIT_TYPES[circuit_type]["function"](num_features, **params).decompose()

    return circuit


def transpile_circuit(seed: int, circuit: QuantumCircuit, backend: Backend, optimization_level: int, initial_layout: list[int] = None) -> QuantumCircuit:

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed,
        initial_layout=initial_layout
    )

    return pm.run(circuit)

def compute_kernel_block(X, kernel, X_support=None):
    """
    Computes the quantum kernel matrix between X and X_support row by row,
    ensuring symmetry when X == X_support.

    Args:
        X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
        kernel (FidelityQuantumKernel): An instance of FidelityQuantumKernel used for evaluation.
        X_support (np.ndarray or None): Optional support feature matrix of shape (m_samples, n_features).
            If None, X is used as the support (i.e., computes a symmetric kernel matrix).

    Returns:
        np.ndarray: Kernel matrix of shape (n_samples, m_samples).
    """

    if X_support is None:
        X_support = X
        symmetric = True
    else:
        symmetric = np.array_equal(X, X_support)

    n, m = len(X), len(X_support)
    K = np.zeros((n, m))

    for i in range(n):
        if i % 25 == 0:
            logger.info(f"Computing kernel row {i+1}/{n}")

        start_j = i if symmetric else 0

        partial_kernel = kernel.evaluate(
            x_vec=[X[i]],
            y_vec=X_support[start_j:]
        )[0] 

        K[i, start_j:] = partial_kernel
        if symmetric:
            K[start_j:, i] = partial_kernel 

    return K


def fit_qsvc_parallel(train_features, train_labels, test_features, sampler, feature_map, pass_manager=None):

    logger.info(f"Inizialising fidelity")
    fidelity = ComputeUncompute(sampler=sampler, pass_manager=pass_manager)

    logger.info(f"Inizialising fidelity quantum kernel")
    kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)  

    n = len(train_features)
    logger.info(f"Sample size: {n}")

    logger.info(f"Numeber of features: {train_features.shape[1]}")

    start_k_train = time.time()
    logger.info(f"Computing quantum kernel (train vs train)")
    K_train = compute_kernel_block(train_features, kernel)
    elapsed_k_train = time.time() - start_k_train

    logger.info(f"Elapsed Time for kernel (train vs train) calculation {elapsed_k_train}")

    start_k_test = time.time()
    logger.info(f"Computing quantum kernel (test vs train)")
    K_test = compute_kernel_block(test_features, kernel, train_features)
    elapsed_k_test = time.time() - start_k_test 

    logger.info(f"Elapsed Time for kernel ((test vs train)) calculation {elapsed_k_test}")

    logger.info(f"Inizialising SVC with precomputed quantum kernel")
    qsvc = SVC(kernel='precomputed',
               verbose=True,
               )

    start = time.time()
    logger.info(f"Running QSVC fit with precomputed quantum kernel")
    qsvc.fit(K_train, train_labels)
    elapsed = time.time() - start 

    logger.info(f"Elapsed Time for QSVC fit with precomputed quantum kernel {elapsed}")

    return {"fit_obj": qsvc, "elapsed_time": elapsed, "kernel": {"kernel_train": K_train, 
                                                                 "elapsed_time_kernel_train": elapsed_k_train, 
                                                                 "kernel_test": K_test, 
                                                                 "elapsed_time_kernel_test": elapsed_k_test
                                                                 }}


def get_optimizer(opt_method: str, opt_params: dict = None) -> object:
    optimizer = OPTIMIZERS[opt_method](**opt_params)
    return optimizer

def fit_vqc(train_features, train_labels, sampler, feature_map, ansatz, optimizer):

    def callback_graph(weights, obj_func_eval):
        clear_output(wait=True)
        objective_func_vals.append(obj_func_eval)

    logger.info(f"Inizialising VQC")
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback_graph
    )
    
    # clear objective value history
    objective_func_vals = []

    start = time.time()
    logger.info(f"Running VQC")
    vqc.fit(train_features, train_labels)
    elapsed = time.time() - start

    return {"fit_obj": vqc, "elapsed_time": elapsed, "objective_func": objective_func_vals}
    
def get_quantum_score(obj, features, labels):
    score = obj.score(features, labels)
    return score


def fit_model_parallel(sampler, pass_manager, method_params, train_features, train_labels, test_features, test_labels):
    """
    Fit a quantum model based on the method parameters provided.
    
    Args:
        train_features (array-like): Training features.
        train_labels (array-like): Training labels.
        sampler (Sampler): Sampler object for quantum computations.
        method_params (dict): Parameters for the quantum method, including 'method_type', 'feature_map', 'ansatz', and 'optimizer'.
    
    Returns:
        tuple: Fitted quantum model and additional information.
    """
    
    method_type = method_params['method_type']
    
    if method_type == 'QSVC':

        logger.info(f"Creating feature map circuit - {method_params['feature_map']}")
        feature_map = get_circuit(train_features.shape[1], 
                                  method_params['feature_map'], 
                                  method_params['feature_map_params'])

        logger.info(f"Starting {method_type} fit")
        model_dict = fit_qsvc_parallel(train_features, train_labels, test_features, sampler, feature_map, pass_manager=pass_manager)
        
        model_dict['feature_map'] = feature_map

        logger.info(f"Calculating train score ...")
        K_train = model_dict['kernel']['kernel_train']
        model_dict["train_score"] = model_dict["fit_obj"].score(K_train, train_labels)
        logger.info(f"Calculating test score ...")
        K_test = model_dict['kernel']['kernel_test']
        model_dict["test_score"] =  model_dict["fit_obj"].score(K_test, test_labels)
        return model_dict
    
    elif method_type == 'VQC':
        logger.info(f"Creating feature map circuit - {method_params['feature_map']}")
        feature_map = get_circuit(train_features.shape[1], method_params['feature_map'], method_params['feature_map_params'])

        logger.info(f"Creating ansatz circuit - {method_params['ansatz']}")
        ansatz = get_circuit(train_features.shape[1], method_params['ansatz'], method_params['ansatz_params'])
        
        logger.info(f"Inizialising Optimizer {method_params['opt_method']}")
        optimizer = get_optimizer(method_params['opt_method'], method_params.get('opt_method_params', {}))
        
        logger.info(f"Starting {method_type} fit")
        model_dict = fit_vqc(train_features, train_labels, sampler, feature_map, ansatz, optimizer)
        model_dict['feature_map'] = feature_map
        model_dict['ansatz'] = ansatz
        
        logger.info(f"Calculating train score ...")
        model_dict["train_score"] = model_dict["fit_obj"].score(train_features, train_labels)
        logger.info(f"Calculating test score ...")
        model_dict["test_score"] =  model_dict["fit_obj"].score(test_features, test_labels)
        return model_dict
    
    else:
        logger.error(f"Unsupported method type: {method_type}")
        raise ValueError(f"Unsupported method type: {method_type}")
        return None
        