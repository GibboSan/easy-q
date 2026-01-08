from typing import List, Optional

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, generate_preset_pass_manager
from qiskit.circuit import ParameterVector, Parameter
from qiskit.providers import Backend

from pipeline.problems.abstract_problem import AbstractProblem


class QAOACircuit:

    def __init__(
        self,
        seed: int,
        problem: AbstractProblem,
        num_qubits: int,
        num_layers: int = 1,
        backend: Optional[Backend] = None
    ):
        self.seed = seed
        self.problem = problem
        self.hamiltonian = problem.hamiltonian
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self._backend = backend

        self.gammas = ParameterVector("gammas", num_layers)
        self.betas = ParameterVector("betas", num_layers)

        self._circuit: Optional[QuantumCircuit] = None
        self._transpiled_circuit: Optional[QuantumCircuit] = None

    @property
    def backend(self) -> Optional[Backend]:
        return self._backend

    @backend.setter
    def backend(self, new_backend: Backend):
        self._backend = new_backend
        self._transpiled_circuit = None

    def _initial_state(self, qreg: QuantumRegister) -> QuantumCircuit:
        qc = QuantumCircuit(qreg)
        qc.h(qreg)
        return qc

    def _cost_operator(self, qc: QuantumCircuit, qreg: QuantumRegister, gamma: Parameter):
        for pauli, coeff in self.hamiltonian.to_list():
            indices = [i for i, p in enumerate(pauli[::-1]) if p == 'Z']
            if not indices:
                continue
            if len(indices) == 1:
                qc.rz(2 * gamma * coeff.real, qreg[indices[0]])
            else:
                qc.rzz(2 * gamma * coeff.real, qreg[indices[0]], qreg[indices[1]])

    def _mixer_operator(self, qc: QuantumCircuit, qreg: QuantumRegister, beta: Parameter):
        for i in range(self.num_qubits):
            qc.rx(2 * beta, qreg[i])

    def build(self) -> QuantumCircuit:

        qreg = QuantumRegister(self.num_qubits, name="quantum_register")
        creg = ClassicalRegister(self.num_qubits, name="classic_register")
        qc = QuantumCircuit(qreg, creg)

        qc.compose(self._initial_state(qreg), qubits=qreg, inplace=True)

        for i in range(self.num_layers):
            self._cost_operator(qc, qreg, self.gammas[i])
            self._mixer_operator(qc, qreg, self.betas[i])

        for i in range(self.num_qubits):
            qc.measure(qreg[i], creg[i])

        self._circuit = qc
        self._transpiled_circuit = None

        return qc

    def get_parameterized_circuit(self) -> QuantumCircuit:
        if self._circuit is None:
            self.build()
        return self._circuit

    def get_bound_circuit(self, gammas: List[float], betas: List[float]) -> QuantumCircuit:
        base = self._transpiled_circuit if self._transpiled_circuit else self._circuit
        if base is None:
            base = self.build()
        param_map = {
            **dict(zip(self.gammas, gammas)),
            **dict(zip(self.betas, betas))
        }
        return base.assign_parameters(param_map, inplace=False)

    def transpile(self, optimization_level: int = 3) -> QuantumCircuit:
        if self._circuit is None:
            self.build()
        if self._backend is None:
            raise ValueError("No backend specified for transpilation.")
        pm = generate_preset_pass_manager(
            backend=self._backend,
            optimization_level=optimization_level,
            seed_transpiler=self.seed
        )
        self._transpiled_circuit = pm.run(self._circuit)
        return self._transpiled_circuit
