from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from qiskit.providers import Backend

from pipeline.problems.abstract_problem import AbstractProblem


class AbstractSolver(ABC):
    def __init__(
        self,
        seed: int,
        output_folder: Optional[str] = None,
        solver_params: Optional[Dict[str, Any]] = None,
    ):
        self.seed = seed
        self.output_folder = output_folder or ""
        self.solver_params = solver_params or {}

    @abstractmethod
    def solve(self, problem: AbstractProblem) -> Dict[str, Any]:
        """
        Run the algorithm.

        Returns
        -------
        dict
            - ``classic_best_bitstring``          - str, best binary solution found
            - ``classic_best_objective``          - float, its objective value
            - ``classic_best_status``              - str, status of the classic solution
            - ``solver_best_bitstring``           - str, best binary solution found by the solver
            - ``solver_best_objective``           - float, its objective value
            - ``classic_walltime``              - float, wall-clock time taken by the classical solver in seconds
            - ``solver_walltime``              - float, total wall-clock time in seconds
        (other solver-specific keys may be included)
        """
        pass
