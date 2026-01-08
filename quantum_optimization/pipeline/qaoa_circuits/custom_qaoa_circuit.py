from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RXXGate, RYYGate
from qiskit.providers import Backend

from pipeline.problems.abstract_problem import AbstractProblem
from pipeline.qaoa_circuits.qaoa_circuit import QAOACircuit
from pipeline.utils import hamming_distance


class CustomQAOACircuit(QAOACircuit):

    def __init__(
            self,
            seed: int,
            problem: AbstractProblem,
            num_qubits: int,
            p: int = 1,
            backend: Optional[Backend] = None
    ):
        super().__init__(seed, problem, num_qubits, p, backend)
        self.all_feasible_bitstrings = problem.all_feasible_bitstrings()

    def _initial_state(self, qreg: QuantumRegister) -> QuantumCircuit:

        valid_states_dec = [int(s, 2) for s in self.all_feasible_bitstrings]

        state_vector = np.zeros(2 ** self.num_qubits, dtype=complex)
        amplitude = 1 / np.sqrt(len(valid_states_dec))
        for idx in valid_states_dec:
            state_vector[idx] = amplitude

        qc_init = QuantumCircuit(qreg)
        qc_init.initialize(state_vector, qreg)

        return qc_init

    def _mixer_operator(self, qc: QuantumCircuit, qreg: QuantumRegister, beta: Parameter):

        valid_states_bin = self.all_feasible_bitstrings
        already_seen_pairs = set()
        already_seen_single = set()

        for i in range(len(valid_states_bin) - 1):

            for j in range(i + 1, len(valid_states_bin)):

                dist, diff = hamming_distance(valid_states_bin[i], valid_states_bin[j])

                if dist == 2:

                    q1 = diff[0]
                    q2 = diff[1]

                    if (q1, q2) not in already_seen_pairs:
                        qc.append(RXXGate(2 * beta), [qreg[q1], qreg[q2]])
                        qc.append(RYYGate(2 * beta), [qreg[q1], qreg[q2]])
                        already_seen_pairs.add((q1, q2))
                        already_seen_single.add(q1)
                        already_seen_single.add(q2)

        for i in range(self.num_qubits):
            if i not in already_seen_single:
                qc.rx(2 * beta, qreg[i])
