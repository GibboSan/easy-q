"""Classical solver that delegates to the CP-SAT solver in AbstractProblem."""

import time
from typing import Any, Dict

from pipeline.problems.abstract_problem import AbstractProblem
from pipeline.solvers.abstract_solver import AbstractSolver


class ClassicalSolver(AbstractSolver):
    """Solve a binary problem using the classical CP-SAT solver.

    Simply calls ``problem.get_best_solution()`` which runs the
    OR-Tools CP-SAT solver under the hood.
    """

    def __init__(self, seed: int):
        super().__init__(seed=seed)

    def solve(self, problem: AbstractProblem) -> Dict[str, Any]:
        tic = time.perf_counter()
        best_bitstring, best_objective = problem.get_best_solution()
        total_time = time.perf_counter() - tic

        return {
            "best_bitstring": best_bitstring,
            "best_objective": best_objective,
            "total_time": total_time,
        }
