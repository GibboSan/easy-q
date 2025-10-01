from typing import Any

from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService


def get_aer_from_backend_GPU(seed: int, backend: Any = None) -> AerSimulator:

    #### NON VIENE DEFINITO IL METHOD --> Default = 'automatic' 
    # Select the simulation method automatically based on the circuit and noise model.'

    kwargs = {
        "seed_simulator": seed,
        "device": "GPU",
        "blocking_enable":True,
        "blocking_qubits": 24
    }

    if backend:
        return AerSimulator.from_backend(backend, **kwargs)

    return AerSimulator(**kwargs)

def get_aer_from_backend_CPU(seed: int, backend: Any = None) -> AerSimulator:

    #### NON VIENE DEFINITO IL METHOD --> Default = 'automatic' 
    # Select the simulation method automatically based on the circuit and noise model.'

    kwargs = {
        # "method": "statevector",
        "seed_simulator": seed
    }

    if backend:
        return AerSimulator.from_backend(backend, **kwargs)

    return AerSimulator(**kwargs)


def get_real_backend() -> BackendV2:

    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.backend("ibm_kyiv")
    print(backend)
    return backend




