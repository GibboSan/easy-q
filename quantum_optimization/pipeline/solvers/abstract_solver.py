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
        pass
