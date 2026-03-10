"""CPLEX-based solver for constrained binary quadratic problems.

Uses IBM's docplex modelling library to construct and solve the full
quadratic program (with constraints).  A full CPLEX licence is needed
for large instances; the community edition bundled with docplex handles
small / moderate ones.

This solver can also be used standalone to solve any ``AbstractProblem``.
"""

import logging
import time
from typing import Any, Dict

from docplex.mp.model import Model as CplexModel
from qiskit.providers import Backend

from pipeline.problems.abstract_problem import AbstractProblem
from pipeline.solvers.abstract_solver import AbstractSolver

logger = logging.getLogger("pipeline_logger")


class CplexSolver(AbstractSolver):
    """Solve a QuadraticProgram directly with CPLEX via docplex.

    Parameters
    ----------
    problem : AbstractProblem
        The optimisation problem.
    backend : Backend
        Unused (present for API consistency with other solvers).
    seed : int
        Random seed forwarded to CPLEX.
    time_limit : float
        Maximum solver wall-time in seconds.
    log_output : bool
        Whether to print CPLEX solver logs.
    """

    def __init__(
        self,
        problem: AbstractProblem,
        backend: Backend,
        seed: int,
        time_limit: float = 300.0,
        log_output: bool = False,
        **kwargs,
    ):
        super().__init__(problem=problem, backend=backend, seed=seed)
        self.time_limit = time_limit
        self.log_output = log_output

    # ------------------------------------------------------------------ #

    def solve(self) -> Dict[str, Any]:
        logger.info("CplexSolver: building docplex model")
        tic = time.perf_counter()

        qp = self.problem.quadratic_binary_problem

        model = CplexModel(name="cplex_solver")
        model.parameters.timelimit = self.time_limit
        model.parameters.randomseed = self.seed % (2**31)
        if not self.log_output:
            model.set_log_output(None)

        var_list = list(qp.variables)

        # --- Variables ----------------------------------------------------
        x_vars = {
            var.name: model.binary_var(name=var.name) for var in var_list
        }

        # --- Objective ----------------------------------------------------
        obj = qp.objective.constant

        for i, coeff in qp.objective.linear.to_dict().items():
            obj += coeff * x_vars[var_list[i].name]

        for (i, j), coeff in qp.objective.quadratic.to_dict().items():
            obj += coeff * x_vars[var_list[i].name] * x_vars[var_list[j].name]

        model.minimize(obj)

        # --- Constraints --------------------------------------------------
        for constraint in qp.linear_constraints:
            lhs = model.sum(
                coeff * x_vars[var_list[i].name]
                for i, coeff in constraint.linear.to_dict().items()
            )
            if constraint.sense == constraint.Sense.EQ:
                model.add_constraint(lhs == constraint.rhs, constraint.name)
            elif constraint.sense == constraint.Sense.LE:
                model.add_constraint(lhs <= constraint.rhs, constraint.name)
            elif constraint.sense == constraint.Sense.GE:
                model.add_constraint(lhs >= constraint.rhs, constraint.name)

        # --- Solve --------------------------------------------------------
        solution = model.solve()
        solve_time = time.perf_counter() - tic

        if solution is None:
            logger.warning("CplexSolver: no feasible solution found")
            return {
                "best_bitstring": None,
                "best_objective": None,
                "solve_time": solve_time,
                "status": "INFEASIBLE",
            }

        bitstring = "".join(
            str(int(round(solution.get_value(x_vars[var.name]))))
            for var in var_list
        )
        objective = float(self.problem.evaluate_cost(bitstring))

        logger.info(
            f"CplexSolver: bitstring={bitstring}, "
            f"objective={objective}, time={solve_time:.2f}s"
        )

        return {
            "best_bitstring": bitstring,
            "best_objective": objective,
            "solve_time": solve_time,
            "status": str(model.solve_details.status),
        }
