from typing import Any

from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError, depolarizing_error, thermal_relaxation_error
from qiskit import QuantumCircuit


def get_aer_from_backend(seed: int, backend: Any = None, noise_model: Any = None, method: str = "automatic") -> AerSimulator:

    kwargs = {
        "seed_simulator": seed,
        "method": method
    }
    
    if backend:
        if noise_model:
            kwargs["noise_model"] = noise_model
        return AerSimulator.from_backend(backend, **kwargs)
    
    if isinstance(noise_model, NoiseModel):
        kwargs["noise_model"] = noise_model
    return AerSimulator(**kwargs)


def get_real_backend(backend_name: str) -> BackendV2:

    # crn = "insert-crn"
    # key = "insert-key"

    service = QiskitRuntimeService(
        # channel="ibm_quantum_platform",
        # instance=crn,
        # token=key
    )

    return service.backend(backend_name)

# SAVE YOUR ACCESS CREDENTIALS for SPECIFIC INSTANCES
# to run previously in the environment.

# from qiskit_ibm_runtime import QiskitRuntimeService
 
# QiskitRuntimeService.save_account(
#   token="<your-api-key>", # Use the 44-character API_KEY you created and saved from the IBM Quantum Platform Home dashboard
#   name="<account-name>", # Optional
#   instance="<IBM Cloud CRN or instance name>", # Optional
#   set_as_default=True, # Optional
#   overwrite=True, # Optional
# )

def build_pruned_noise_model(real_backend, transpiled_circuit) -> NoiseModel:

    active_qubits = sorted({
        transpiled_circuit.find_bit(q).index
        for instr in transpiled_circuit.data
        for q in instr.qubits
    })
    active_set = set(active_qubits)

    nm = NoiseModel()
    target = real_backend.target

    if "measure" in target:
        for q in active_qubits:
            if (q,) in target["measure"]:
                err = target["measure"][(q,)].error
                if err is None:
                    continue
                if isinstance(err, float):
                    re = ReadoutError([[1 - err, err], [err, 1 - err]])
                    nm.add_readout_error(re, [q])
                else:
                    nm.add_quantum_error(err, "measure", [q])

    added = set()

    for op_name in target:
        if op_name in ["measure", "reset", "delay", "barrier", "if_else"]:
            continue
        for phys_qubits in target[op_name]:
            if not isinstance(phys_qubits, tuple):
                continue
            if not all(q in active_set for q in phys_qubits):
                continue
            key = (op_name, phys_qubits)
            if key in added:
                continue
            err = target[op_name][phys_qubits].error
            if err is None:
                continue
            if isinstance(err, float):
                dep_err = depolarizing_error(err, len(phys_qubits))
                nm.add_quantum_error(dep_err, op_name, phys_qubits)
            else:
                nm.add_quantum_error(err, op_name, phys_qubits)
            added.add(key)

    return nm

