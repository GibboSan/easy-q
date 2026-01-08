
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization.converters import QuadraticProgramToQubo

from pipeline.problems.qgs_problem import QGSProblem


class QGSUncProblem(QGSProblem):

    def build_hamiltonian(self) -> SparsePauliOp:
        qubo = QuadraticProgramToQubo(penalty=0.0).convert(self._quadratic_problem)
        hamiltonian, _ = qubo.to_ising()
        return hamiltonian
