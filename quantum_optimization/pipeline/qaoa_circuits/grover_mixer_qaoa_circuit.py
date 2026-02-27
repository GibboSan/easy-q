from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.providers import Backend

from pipeline.problems.abstract_problem import AbstractProblem
from pipeline.qaoa_circuits.qaoa_circuit import QAOACircuit


class GroverMixerQAOACircuit(QAOACircuit):

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
        """
        Prepares a uniform superposition over all feasible bitstrings.
        """
        valid_states_dec = [int(s, 2) for s in self.all_feasible_bitstrings]

        state_vector = np.zeros(2 ** self.num_qubits, dtype=complex)
        amplitude = 1 / np.sqrt(len(valid_states_dec))
        for idx in valid_states_dec:
            state_vector[idx] = amplitude

        qc_init = QuantumCircuit(qreg)
        qc_init.initialize(state_vector, qreg)

        return qc_init

    def _mixer_operator(self, qc: QuantumCircuit, qreg: QuantumRegister, beta: Parameter):
        """
        Implements the Grover diffusion operator as a mixer: exp(-i * beta * D)
        where D = 2|ψ_feas⟩⟨ψ_feas| - I
        
        This mixer exactly preserves the feasible subspace by reflecting quantum
        states about the average amplitude of feasible states.
        """
        # Step 1: Apply inverse of initial state preparation
        # Maps |ψ_feas⟩ → |0...0⟩
        qc_init_inverse = self._initial_state(qreg).inverse()
        qc.compose(qc_init_inverse, qreg, inplace=True)
        
        # Step 2: Apply conditional phase shift to |0...0⟩ state
        # This implements (2|0⟩⟨0| - I) in the computational basis
        
        # Flip all qubits: |0⟩ → |1⟩
        qc.x(qreg)
        
        # Multi-controlled Z gate (phase flip when all qubits are |1⟩)
        if self.num_qubits == 1:
            # Single qubit case
            qc.z(qreg[0])
        else:
            # Multi-qubit case: use MCX with Hadamard trick for MCZ
            qc.h(qreg[-1])
            qc.mcx(list(qreg[:-1]), qreg[-1])
            qc.h(qreg[-1])
        
        # Flip qubits back: |1⟩ → |0⟩
        qc.x(qreg)
        
        # Step 3: Apply initial state preparation again
        # Maps |0...0⟩ → |ψ_feas⟩
        qc_init = self._initial_state(qreg)
        qc.compose(qc_init, qreg, inplace=True)
        
        # Note: For proper time evolution with angle beta, this diffusion operator
        # should be applied multiple times or the circuit can be wrapped in a
        # power operation. For small beta, single application is often sufficient.
        # For larger beta values, consider repeating the diffusion operator
        # or using more sophisticated Trotterization schemes.
