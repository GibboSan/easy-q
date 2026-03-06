"""
Quantum Frank-Wolfe Solver
==========================

Implements Algorithm 1 from:

    Yurtsever et al., *"Q-FW: A Hybrid Classical-Quantum Frank-Wolfe
    for Quadratic Binary Optimization"* (2022).

The Adiabatic Quantum Computing (AQC) step of the original paper is
replaced with **QAOA**, using the circuits available in
``pipeline.qaoa_circuits``.

Typical usage::

    from pipeline.solvers.frank_wolfe import QFWSolver
    from pipeline.problems.abstract_problem import AbstractProblem
    from pipeline.qaoa_circuits.qaoa_circuit import QAOACircuit
    from pipeline.backends import get_aer_from_backend

    problem = AbstractProblem(seed=0, problem_params={...})
    backend = get_aer_from_backend(seed=0)

    solver = QFWSolver(
        problem=problem,
        backend=backend,
        seed=0,
        circuit_class=QAOACircuit,
        num_fw_iterations=10,
        lmo_params={"num_layers": 1, "num_starting_points": 3},
    )
    result = solver.solve()
"""

import logging
import time
from typing import Dict, Optional, Type

import numpy as np
from qiskit.providers import Backend

from pipeline.problems.abstract_problem import AbstractProblem
from pipeline.qaoa_circuits.qaoa_circuit import QAOACircuit
from pipeline.solvers.abstract_solver import AbstractSolver

from pipeline.solvers.frank_wolfe.gradient import build_Q_and_c, compute_gradient
from pipeline.solvers.frank_wolfe.relaxation import build_initial_point
from pipeline.solvers.frank_wolfe.linear_minimisation import qaoa_lmo
from pipeline.solvers.frank_wolfe.rounding import round_solution
from pipeline.solvers.frank_wolfe.qfw_utils import (
    step_size,
    frank_wolfe_gap,
    evaluate_continuous_objective,
    ConvergenceTracker,
)

logger = logging.getLogger("pipeline_logger")


class QFWSolver(AbstractSolver):
    """
    Quantum Frank-Wolfe solver for Quadratic Binary Optimisation.

    Parameters
    ----------
    problem : AbstractProblem
        The optimisation problem to solve.
    backend : Backend
        Quantum backend (real or simulator).
    seed : int
        Random seed.
    circuit_class : Type[QAOACircuit]
        QAOA circuit variant used by the LMO
        (``QAOACircuit``, ``AncillaQAOACircuit``, …).
    num_fw_iterations : int
        Maximum number of Frank-Wolfe outer iterations (*T*).
    step_size_rule : str
        ``'standard'`` (2/(t+2)), ``'diminishing'``, or ``'constant'``.
    rounding_method : str
        ``'threshold'``, ``'randomized'``, or ``'greedy'``.
    convergence_tol : float
        Stop early when the FW gap drops below this value.
    lmo_params : dict, optional
        Keyword arguments forwarded to :func:`qaoa_lmo`:
        ``num_layers``, ``num_starting_points``, ``bounds``,
        ``optimization_params``, ``num_estimator_shots``,
        ``num_sampler_shots``, ``use_cache``, …
    """

    def __init__(
        self,
        problem: AbstractProblem,
        backend: Backend,
        seed: int,
        circuit_class: Type[QAOACircuit],
        num_fw_iterations: int = 10,
        step_size_rule: str = "standard",
        rounding_method: str = "greedy",
        convergence_tol: float = 1e-6,
        lmo_params: Optional[dict] = None,
    ):
        super().__init__(
            problem=problem,
            backend=backend,
            seed=seed,
            solver_params={
                "num_fw_iterations": num_fw_iterations,
                "step_size_rule": step_size_rule,
                "rounding_method": rounding_method,
                "convergence_tol": convergence_tol,
                "lmo_params": lmo_params or {},
            },
        )
        self.circuit_class = circuit_class
        self.num_fw_iterations = num_fw_iterations
        self.step_size_rule = step_size_rule
        self.rounding_method = rounding_method
        self.convergence_tol = convergence_tol
        self.lmo_params = lmo_params or {}

        # Extract objective coefficients
        self.Q, self.c, self.offset = build_Q_and_c(problem)
        self.n = len(self.c)

        # Convergence tracker
        self.tracker = ConvergenceTracker()

    # --------------------------------------------------------------------- #
    #  Main loop
    # --------------------------------------------------------------------- #

    def solve(self) -> Dict:
        """
        Run the Quantum Frank-Wolfe algorithm.

        Returns
        -------
        dict
            ``best_bitstring``          - str, best binary solution found
            ``best_objective``          - float, its objective value
            ``continuous_solution``     - list, final x in [0,1]^n
            ``continuous_objective``    - float, f(x_T)
            ``rounding_method``         - str
            ``num_iterations``          - int
            ``convergence``             - dict from ConvergenceTracker
            ``total_time``              - float (seconds)
            ``classic_best_bitstring``  - str
            ``classic_best_objective``  - float
            ``seed``                    - int
        """
        logger.info(
            f"Starting Q-FW with T={self.num_fw_iterations}, "
            f"step_size={self.step_size_rule}, "
            f"rounding={self.rounding_method}"
        )

        tic_total = time.perf_counter()

        # ---- initialise ----------------------------------------------------
        x = build_initial_point(self.problem)
        logger.info(
            "Initial point: f(x0) = "
            f"{evaluate_continuous_objective(self.Q, self.c, x, self.offset):.6f}"
        )

        # ---- FW loop -------------------------------------------------------
        for t in range(self.num_fw_iterations):
            logger.info(f"--- FW iteration {t + 1}/{self.num_fw_iterations} ---")

            # 1. Gradient
            grad = compute_gradient(self.Q, self.c, x)

            # 2. LMO (QAOA replaces AQC)
            lmo_result = qaoa_lmo(
                grad=grad,
                problem=self.problem,
                circuit_class=self.circuit_class,
                backend=self.backend,
                seed=self.seed + t,          # vary seed per iteration
                **self.lmo_params,
            )
            s = lmo_result["vertex"]

            # 3. FW gap
            gap = frank_wolfe_gap(grad, x, s)

            # 4. Step size
            gamma = step_size(t, self.step_size_rule)

            # 5. Convex update
            x = (1 - gamma) * x + gamma * s

            # 6. Record
            obj = evaluate_continuous_objective(self.Q, self.c, x, self.offset)
            self.tracker.record(
                objective=obj,
                fw_gap=gap,
                gamma=gamma,
                lmo_objective=lmo_result["lmo_objective"],
                lmo_time=lmo_result["lmo_time"],
                vertex=s,
                iterate=x,
            )

            logger.info(
                f"  f(x)={obj:.6f}  gap={gap:.6f}  "
                f"γ={gamma:.4f}  LMO={lmo_result['lmo_time']:.2f}s"
            )

            # 7. Early stopping
            if gap < self.convergence_tol:
                logger.info(
                    f"Converged at iteration {t + 1} "
                    f"(gap={gap:.2e} < tol={self.convergence_tol:.2e})"
                )
                break

        # ---- rounding ------------------------------------------------------
        logger.info(f"Rounding with method='{self.rounding_method}'")
        best_bitstring, best_objective = round_solution(
            x, self.problem, method=self.rounding_method, seed=self.seed
        )

        # Also check all LMO vertices (they are feasible binary vectors)
        for vert in self.tracker.vertices:
            bs = "".join(str(int(v)) for v in vert)
            if self.problem.is_feasible(bs)[0]:
                cost = self.problem.evaluate_cost(bs)
                if cost < best_objective:
                    logger.info(
                        f"Vertex {bs} (cost={cost}) beats rounded solution "
                        f"(cost={best_objective})"
                    )
                    best_bitstring = bs
                    best_objective = cost

        total_time = time.perf_counter() - tic_total

        classic_best = self.problem.get_best_solution()
        logger.info(
            f"Q-FW  → bitstring={best_bitstring}, objective={best_objective}"
        )
        logger.info(
            f"Classic → bitstring={classic_best[0]}, objective={classic_best[1]}"
        )
        logger.info(f"Total Q-FW time: {total_time:.2f}s")

        return {
            "best_bitstring": best_bitstring,
            "best_objective": best_objective,
            "continuous_solution": x.tolist(),
            "continuous_objective": float(
                evaluate_continuous_objective(self.Q, self.c, x, self.offset)
            ),
            "rounding_method": self.rounding_method,
            "num_iterations": len(self.tracker.objective_values),
            "convergence": self.tracker.summary(),
            "total_time": total_time,
            "classic_best_bitstring": classic_best[0],
            "classic_best_objective": classic_best[1],
            "seed": self.seed,
        }
