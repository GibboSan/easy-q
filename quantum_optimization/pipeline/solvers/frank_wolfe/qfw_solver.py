"""Faithful FWAL solver in lifted CP space.

This implementation follows the Section 3--4 algorithmic structure from
Yurtsever et al. (2022): lifted model construction, FWAL primal-dual
iterations, and Section-4 style rounding from W_hat.
"""

import logging
import time
from typing import Dict, Optional, Type

from qiskit.providers import Backend

from pipeline.problems.abstract_problem import AbstractProblem
from pipeline.qaoa_circuits.qaoa_circuit import QAOACircuit
from pipeline.solvers.abstract_solver import AbstractSolver

from pipeline.solvers.frank_wolfe.gradient import (
    build_cp_affine_model,
    apply_affine_map,
    compute_fwal_gradient,
    evaluate_cp_objective,
)
from pipeline.solvers.frank_wolfe.relaxation import (
    build_initial_primal_matrix,
    build_initial_dual_vector,
    symmetrise,
)
from pipeline.solvers.frank_wolfe.linear_minimisation import fwal_lmo
from pipeline.solvers.frank_wolfe.rounding import round_from_W
from pipeline.solvers.frank_wolfe.qfw_utils import (
    primal_step_size,
    penalty_parameter,
    dual_step_size,
    fw_gap_matrix,
    residual_norm,
    ConvergenceTracker,
)

logger = logging.getLogger("pipeline_logger")


class QFWSolver(AbstractSolver):
    """
    Quantum Frank-Wolfe Augmented Lagrangian (FWAL) solver.

    Parameters
    ----------
    problem : AbstractProblem
        The optimisation problem to solve.
    backend : Backend
        Quantum backend (real or simulator).
    seed : int
        Random seed.
    circuit_class : Type[QAOACircuit]
        Kept for API compatibility. Not used by the current classical
        FWAL-LMO backend.
    num_fw_iterations : int
        Maximum number of Frank-Wolfe outer iterations (*T*).
    beta0 : float
        Initial penalty parameter for FWAL.
    dual_step_rule : str
        Dual ascent step-size rule (currently ``'constant'``).
    rounding_project : bool
        Whether to apply projection in Section-4 style rounding.
    convergence_tol : float
        Early stop threshold for both residual norm and FW gap.
    lmo_params : dict, optional
        Keyword arguments forwarded to :func:`fwal_lmo`.
    """

    def __init__(
        self,
        problem: AbstractProblem,
        backend: Backend,
        seed: int,
        circuit_class: Type[QAOACircuit],
        num_fw_iterations: int = 10,
        beta0: float = 1.0,
        dual_step_rule: str = "constant",
        rounding_project: bool = True,
        convergence_tol: float = 1e-6,
        lmo_params: Optional[dict] = None,
    ):
        super().__init__(
            problem=problem,
            backend=backend,
            seed=seed,
            solver_params={
                "num_fw_iterations": num_fw_iterations,
                "beta0": beta0,
                "dual_step_rule": dual_step_rule,
                "rounding_project": rounding_project,
                "convergence_tol": convergence_tol,
                "lmo_params": lmo_params or {},
            },
        )
        self.circuit_class = circuit_class
        self.num_fw_iterations = num_fw_iterations
        self.beta0 = beta0
        self.dual_step_rule = dual_step_rule
        self.rounding_project = rounding_project
        self.convergence_tol = convergence_tol
        self.lmo_params = lmo_params or {}

        self.model = build_cp_affine_model(problem)
        self.n = self.model.n
        self.p = self.model.p
        self.d = len(self.model.rhs)

        # Convergence tracker
        self.tracker = ConvergenceTracker()

    # --------------------------------------------------------------------- #
    #  Main loop
    # --------------------------------------------------------------------- #

    def solve(self) -> Dict:
        """
        Run the FWAL algorithm in lifted CP space.

        Returns
        -------
        dict
            ``best_bitstring``          - str, best binary solution found
            ``best_objective``          - float, its objective value
            ``continuous_solution``     - list, decoded x_hat from W_T
            ``continuous_objective``    - float, objective of rounded solution
            ``rounding_method``         - str (always ``section4_rank1``)
            ``num_iterations``          - int
            ``convergence``             - dict from ConvergenceTracker
            ``total_time``              - float (seconds)
            ``classic_best_bitstring``  - str
            ``classic_best_objective``  - float
            ``seed``                    - int
        """
        logger.info(
            f"Starting Q-FW with T={self.num_fw_iterations}, "
            f"beta0={self.beta0}, dual_step={self.dual_step_rule}"
        )

        tic_total = time.perf_counter()

        # ---- initialise ----------------------------------------------------
        W = build_initial_primal_matrix(self.p, mode="zeros")
        y = build_initial_dual_vector(self.d)
        logger.info(
            "Initial state: "
            f"Tr(CW0)={evaluate_cp_objective(self.model.C, W):.6f}"
        )

        # ---- FW loop -------------------------------------------------------
        for t in range(1, self.num_fw_iterations + 1):
            logger.info(f"--- FW iteration {t}/{self.num_fw_iterations} ---")

            beta_t = penalty_parameter(self.beta0, t)
            gamma_t = dual_step_size(self.beta0, self.dual_step_rule)

            # 1. Primal gradient matrix G_t
            G = compute_fwal_gradient(
                self.model.C,
                self.model.constraint_matrices,
                self.model.rhs,
                W,
                y,
                beta_t,
            )

            # 2. LMO over Delta^p extreme points: H_t = w w^T
            lmo_result = fwal_lmo(
                G=G,
                seed=self.seed + t,
                **self.lmo_params,
            )
            H = lmo_result["vertex_matrix"]

            # 3. FW gap at current iterate
            gap = fw_gap_matrix(G, W, H)

            # 4. Primal update W_{t+1}
            eta_t = primal_step_size(t)
            W = (1.0 - eta_t) * W + eta_t * H
            W = symmetrise(W)

            # 5. Dual update y_{t+1}
            residual = apply_affine_map(self.model.constraint_matrices, W) - self.model.rhs
            y = y + gamma_t * residual

            # 6. Record
            cp_obj = evaluate_cp_objective(self.model.C, W)
            res_norm = residual_norm(residual)
            self.tracker.record(
                cp_objective=cp_obj,
                fw_gap=gap,
                residual=res_norm,
                eta=eta_t,
                beta=beta_t,
                gamma=gamma_t,
                lmo_objective=lmo_result["lmo_objective"],
                lmo_time=lmo_result["lmo_time"],
                lmo_bitstring=lmo_result["bitstring"],
            )

            logger.info(
                f"  Tr(CW)={cp_obj:.6f}  gap={gap:.6f}  "
                f"res={res_norm:.3e}  eta={eta_t:.4f}  beta={beta_t:.4f}"
            )

            # 7. Early stopping
            if gap < self.convergence_tol and res_norm < self.convergence_tol:
                logger.info(
                    f"Converged at iteration {t} "
                    f"(gap={gap:.2e}, residual={res_norm:.2e}, tol={self.convergence_tol:.2e})"
                )
                break

        # ---- rounding ------------------------------------------------------
        logger.info("Rounding with Section-4 rank-1 decode")
        rounding = round_from_W(W, self.problem, project=self.rounding_project)
        best_bitstring = rounding["bitstring"]
        best_objective = rounding["objective"]
        x_hat = rounding["x_hat"]
        X_hat = rounding["X_hat"]

        # Also inspect LMO vertices mapped to x = w[1:]
        for lmo_bs in self.tracker.lmo_bitstrings:
            bs = lmo_bs[1:]
            if len(bs) != self.n:
                continue
            if self.problem.is_feasible(bs)[0]:
                cost = self.problem.evaluate_cost(bs)
                if cost < best_objective:
                    logger.info(
                        f"Vertex {bs} (cost={cost}) beats rounded solution "
                        f"(cost={best_objective})"
                    )
                    best_bitstring = bs
                    best_objective = float(cost)

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
            "continuous_solution": x_hat.tolist(),
            "continuous_objective": float(best_objective),
            "rounding_method": "section4_rank1",
            "num_iterations": len(self.tracker.cp_objective_values),
            "convergence": self.tracker.summary(),
            "total_time": total_time,
            "classic_best_bitstring": classic_best[0],
            "classic_best_objective": classic_best[1],
            "W_solution": W.tolist(),
            "X_solution": X_hat.tolist(),
            "dual_solution": y.tolist(),
            "residual_norm": self.tracker.residual_norms[-1] if self.tracker.residual_norms else None,
            "seed": self.seed,
        }
