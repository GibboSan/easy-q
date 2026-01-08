import itertools
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.quantum_info import SparsePauliOp

from pipeline.problems.abstract_problem import AbstractProblem
from pipeline.qaoa_circuits.qaoa_circuit import QAOACircuit


class AncillaQAOACircuit(QAOACircuit):

    def __init__(
        self,
        seed: int,
        problem: AbstractProblem,
        num_qubits: int,
        num_layers: int = 1,
        backend: Optional[Backend] = None
    ):
        super().__init__(seed, problem, num_qubits, num_layers, backend)
        self.qubit_subsets = self.problem.get_qubit_subsets_from_sum1_constraints()
        self.uncontrolled_qubit_indices = [i for i in range(self.num_qubits) if not any(i in subset for subset in self.qubit_subsets)]
        self.n_ancillas = len(self.qubit_subsets)
        self.extend_hamiltonian_with_identity()

    def extend_hamiltonian_with_identity(self):

        extended_op_list = []

        ancilla_indices = [self.num_qubits + i for i in range(self.n_ancillas)]

        for pauli_str, coeff in self.hamiltonian.to_list():

            pauli_str = pauli_str[::-1]
            for index in ancilla_indices:
                pauli_str = pauli_str[:index] + 'I' + pauli_str[index:]
            pauli_str = pauli_str[::-1]

            extended_op_list.append((pauli_str, coeff))

        self.hamiltonian = SparsePauliOp.from_list(extended_op_list)

    def _initial_state(self, qreg: QuantumRegister) -> QuantumCircuit:

        qc_init = QuantumCircuit(self.num_qubits + self.n_ancillas)

        for i in self.uncontrolled_qubit_indices:
            qc_init.h(i)

        counter = 0
        for constraint_qubits in self.qubit_subsets:

            ancilla = self.num_qubits + counter
            k = len(constraint_qubits)

            if k < 2:
                raise ValueError("Each subset of constraint qubits must have at least 2 qubits")

            qc_init.ry(2 * np.arcsin(np.sqrt(1 / k)), constraint_qubits[0])

            qc_init.x(ancilla)

            qc_init.cx(constraint_qubits[0], ancilla)

            for i in range(1, k - 1):
                ctrl = constraint_qubits[i]
                theta = 2 * np.arcsin(np.sqrt(1 / (k - i)))
                qc_init.cry(theta, ancilla, ctrl)
                qc_init.cx(ctrl, ancilla)

            qc_init.cx(ancilla, constraint_qubits[-1])

            for i in range(k - 2, -1, -1):
                ctrl = constraint_qubits[i]
                qc_init.cx(ctrl, ancilla)

            counter += 1

        return qc_init

    def _mixer_operator(self, qc: QuantumCircuit, qreg: QuantumRegister, beta: Parameter):

        for constraint_qubits in self.qubit_subsets:
            for i, j in itertools.combinations(constraint_qubits, 2):
                qc.rxx(2 * beta, i, j)
                qc.ryy(2 * beta, i, j)

        for i in self.uncontrolled_qubit_indices:
            qc.rx(2 * beta, qreg[i])

    def build(self) -> QuantumCircuit:

        qreg = QuantumRegister(self.num_qubits + self.n_ancillas, 'quantum_register')
        creg = ClassicalRegister(self.num_qubits, 'classic_register')
        qc = QuantumCircuit(qreg, creg)

        qc.compose(self._initial_state(qreg), inplace=True)

        for i in range(self.num_layers):
            self._cost_operator(qc, qreg, self.gammas[i])
            self._mixer_operator(qc, qreg, self.betas[i])

        for i in range(self.num_qubits):
            qc.measure(i, i)

        self._circuit = qc
        self._transpiled_circuit = None

        return qc
