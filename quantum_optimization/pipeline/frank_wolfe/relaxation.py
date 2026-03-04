"""
Continuous-relaxation utilities for the Quantum Frank-Wolfe algorithm.

* Maps a gradient vector to a *linear* (diagonal) Hamiltonian that the
  QAOA-based LMO must minimise.
* Provides ``LinearSubproblem``, a lightweight wrapper that presents the
  linear Hamiltonian to any QAOA circuit while delegating constraint
  information to the original problem.
* Builds an initial feasible point for the Frank-Wolfe iteration.
"""

from typing import List, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from pipeline.problems.abstract_problem import AbstractProblem


# ---------------------------------------------------------------------------
#  Hamiltonian from gradient
# ---------------------------------------------------------------------------

def gradient_to_hamiltonian(grad: np.ndarray) -> SparsePauliOp:
    """
    Build a linear (diagonal) Hamiltonian from a gradient vector.

    The LMO minimises::

        min_{s in C}  grad^T s        (s binary)

    Using the mapping  s_i = (I - Z_i) / 2:

        grad^T s  =  sum_i  grad_i / 2  -  sum_i (grad_i / 2) Z_i

    Dropping the constant (it does not affect the ground state):

        H  =  -(1/2) sum_i  grad_i  Z_i

    Parameters
    ----------
    grad : np.ndarray, shape (n,)
        Gradient vector at the current iterate.

    Returns
    -------
    SparsePauliOp
        Linear Hamiltonian whose ground state minimises ``grad^T s``.
    """
    n = len(grad)
    pauli_list = []

    for i in range(n):
        if abs(grad[i]) < 1e-15:
            continue
        # Qiskit Pauli strings are big-endian: rightmost char -> qubit 0
        pauli_chars = ["I"] * n
        pauli_chars[n - 1 - i] = "Z"
        pauli_list.append(("".join(pauli_chars), -0.5 * grad[i]))

    if not pauli_list:
        # Zero gradient → trivial Hamiltonian
        pauli_list = [("I" * n, 0.0)]

    return SparsePauliOp.from_list(pauli_list).simplify()


# ---------------------------------------------------------------------------
#  Lightweight problem wrapper for the LMO
# ---------------------------------------------------------------------------

class LinearSubproblem:
    """
    Presents a *linear* Hamiltonian to QAOA circuits while delegating
    constraint information (feasible bitstrings, sum-1 groups, …) to
    the original :class:`AbstractProblem`.

    This allows constraint-aware variants (``AncillaQAOACircuit``,
    ``CustomQAOACircuit``, ``GroverMixerQAOACircuit``) to work
    unchanged with the Frank-Wolfe linear sub-problem.

    Parameters
    ----------
    original_problem : AbstractProblem
        The full problem (provides constraints, feasibility checks, etc.).
    hamiltonian : SparsePauliOp
        The linear Hamiltonian built from the current FW gradient.
    """

    def __init__(self, original_problem: AbstractProblem, hamiltonian: SparsePauliOp):
        self._original = original_problem
        self.hamiltonian = hamiltonian
        self.seed = original_problem.seed
        self.problem_params = original_problem.problem_params
        self.constraints_sum_1 = original_problem.constraints_sum_1
        self._quadratic_binary_problem = original_problem.quadratic_binary_problem

    # -- delegated properties / methods ------------------------------------

    @property
    def quadratic_binary_problem(self):
        return self._original.quadratic_binary_problem

    def all_feasible_bitstrings(self) -> List[str]:
        return self._original.all_feasible_bitstrings()

    def get_qubit_subsets_from_sum1_constraints(self) -> List[List[int]]:
        return self._original.get_qubit_subsets_from_sum1_constraints()

    def get_feasible_logic_expression_and_total_from_sum1_constraints(self) -> Tuple[str, int]:
        return self._original.get_feasible_logic_expression_and_total_from_sum1_constraints()

    def evaluate_cost(self, solution: str) -> float:
        return self._original.evaluate_cost(solution)

    def is_feasible(self, solution: str, verbose: bool = False) -> Tuple[bool, dict]:
        return self._original.is_feasible(solution, verbose)

    def get_best_solution(self) -> Tuple[str, float]:
        return self._original.get_best_solution()


# ---------------------------------------------------------------------------
#  Initial feasible point
# ---------------------------------------------------------------------------

def build_initial_point(problem: AbstractProblem) -> np.ndarray:
    """
    Build an initial feasible continuous point for Frank-Wolfe.

    * For each sum-1 equality-constraint group of *k* variables,
      sets every variable in the group to 1/k (uniform relaxation).
    * All other (unconstrained) binary variables are set to 0.5
      (midpoint of [0,1]).

    Parameters
    ----------
    problem : AbstractProblem

    Returns
    -------
    x0 : np.ndarray, shape (n,)
        Feasible point in [0,1]^n.
    """
    n = problem.hamiltonian.num_qubits
    x0 = np.full(n, 0.5)

    if problem.constraints_sum_1 is not None:
        constrained_indices: set = set()
        for constraint in problem.constraints_sum_1:
            indices = list(constraint.linear.to_dict().keys())
            k = len(indices)
            for idx in indices:
                x0[idx] = 1.0 / k
                constrained_indices.add(idx)

    return x0
