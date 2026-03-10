from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from qiskit.providers import Backend

from pipeline.problems.abstract_problem import AbstractProblem


class AbstractSolver(ABC):
    def __init__(
        self,
        problem: AbstractProblem,
        backend: Backend,
        seed: int,
        solver_params: Optional[Dict[str, Any]] = None,
    ):
        self.problem = problem
        self.backend = backend
        self.seed = seed
        self.solver_params = solver_params or {}

    @abstractmethod
    def solve(self) -> Dict[str, Any]:
        """
        Run the algorithm.

        Returns
        -------
        dict
            ``best_bitstring``          - str, best binary solution found
            ``best_objective``          - float, its objective value
            ``total_time``              - float, total wall-clock time in seconds
        (other solver-specific keys may be included)
        """
        pass
