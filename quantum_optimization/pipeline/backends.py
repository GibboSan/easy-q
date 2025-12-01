from typing import Any

from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService


def get_aer_from_backend(seed: int, backend: Any = None) -> AerSimulator:

    kwargs = {
        "seed_simulator": seed
    }

    if backend:
        return AerSimulator.from_backend(backend, **kwargs)

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





