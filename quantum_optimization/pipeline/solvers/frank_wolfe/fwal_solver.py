"""Quantum Frank-Wolfe Augmented Lagrangian (Q-FWAL) solver.

Solves constrained quadratic binary optimisation problems via the FWAL
method in lifted copositive space, as described in:

    Yurtsever, Birdal & Golyanik (2022), *Q-FW: A Hybrid
    Classical-Quantum Frank-Wolfe for Quadratic Binary Optimization*,
    ECCV 2022, LNCS 13683, pp. 352-369.

The algorithm:

1. Lifts the constrained QBO into a compact copositive program (CP).
2. Iteratively updates a primal matrix W and dual vector y using the
   Frank-Wolfe + augmented-Lagrangian framework.
3. At each iteration the Linear Minimisation Oracle (LMO) solves an
   unconstrained QUBO over the extreme points of Delta^p.
4. After convergence (or reaching the iteration budget), a rank-1
   rounding step decodes a binary solution for the original problem.

Inequality constraints (<= / >=) are handled transparently via binary
slack variables (see ``gradient.build_cp_affine_model``).

The inner LMO calls existing solvers, including QAOASolver or ClassicalSolver.
"""

import logging
import time
from typing import Dict

from pipeline.problems.abstract_problem import AbstractProblem
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
from quantum_optimization.pipeline.solvers.frank_wolfe.fwal_utils import (
    primal_step_size,
    penalty_parameter,
    dual_step_size,
    fw_gap_matrix,
    residual_norm,
    ConvergenceTracker,
)

logger = logging.getLogger("pipeline_logger")


class FWALSolver(AbstractSolver):
    """Frank-Wolfe Augmented Lagrangian (FWAL) solver.

    Parameters
    ----------
    seed : int
        Random seed.
    lmo_solver : AbstractSolver
        Solver used for the LMO sub-problem at each FW iteration
        (e.g. ``ClassicalSolver`` or ``QAOASolver``).
    num_fw_iterations : int
        Maximum Frank-Wolfe outer iterations (*T*).
    beta0 : float
        Initial penalty parameter for FWAL.
    dual_step_rule : str
        Dual ascent step-size rule (currently ``'constant'``).
    rounding_project : bool
        Whether to apply constraint-aware projection during rounding.
    convergence_tol : float
        Early-stop threshold for both FW gap and residual norm.
    """

    def __init__(
        self,
        seed: int,
        lmo_solver: AbstractSolver,
        num_fw_iterations: int = 10,
        beta0: float = 1.0,
        dual_step_rule: str = "constant",
        rounding_project: bool = True,
        convergence_tol: float = 1e-6,
    ):
        super().__init__(
            seed=seed,
            solver_params={
                "num_fw_iterations": num_fw_iterations,
                "beta0": beta0,
                "dual_step_rule": dual_step_rule,
                "rounding_project": rounding_project,
                "convergence_tol": convergence_tol,
            },
        )
        self.lmo_solver = lmo_solver
        self.num_fw_iterations = num_fw_iterations
        self.beta0 = beta0
        self.dual_step_rule = dual_step_rule
        self.rounding_project = rounding_project
        self.convergence_tol = convergence_tol

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def solve(self, problem: AbstractProblem) -> Dict:
        """Run the FWAL algorithm in lifted CP space.

        Returns
        -------
        dict
            Keys include ``best_bitstring``, ``best_objective``,
            ``continuous_solution``, ``num_iterations``, ``convergence``,
            ``total_time``, and solver diagnostics.
        """
        # Build lifted CP model (handles inequality → slack expansion)
        model = build_cp_affine_model(problem)
        n_original = model.n_original
        n_expanded = model.n_expanded
        p = model.p
        d = len(model.rhs)
        tracker = ConvergenceTracker()

        logger.info(
            f"FWALSolver:/n"
            f"T={self.num_fw_iterations}  "
            f"beta0={self.beta0}  dual_step={self.dual_step_rule}  "
            f"n_orig={n_original}  n_exp={n_expanded}  "
            f"p={p}  d={d}"
            f"LMO solver={type(self.lmo_solver).__name__}"
        )

        tic_total = time.perf_counter()

        # ---- initialise --------------------------------------------------
        W = build_initial_primal_matrix(p, mode="zeros")
        y = build_initial_dual_vector(d)
        logger.info(
            f"Initial Tr(CW0) = {evaluate_cp_objective(model.C, W):.6f}"
        )

        # ---- FW loop -----------------------------------------------------
        for t in range(1, self.num_fw_iterations + 1):
            logger.info(f"--- FW iteration {t}/{self.num_fw_iterations} ---")

            beta_t = penalty_parameter(self.beta0, t)
            gamma_t = dual_step_size(self.beta0, self.dual_step_rule)

            # 1. Gradient
            G = compute_fwal_gradient(
                model.C,
                model.constraint_matrices,
                model.rhs,
                W,
                y,
                beta_t,
            )

            # 2. LMO
            lmo_result = fwal_lmo(
                G,
                seed=self.seed + t,
                solver=self.lmo_solver,
            )
            H = lmo_result["vertex_matrix"]

            # 3. FW gap
            gap = fw_gap_matrix(G, W, H)

            # 4. Primal update
            eta_t = primal_step_size(t)
            W = (1.0 - eta_t) * W + eta_t * H
            W = symmetrise(W)

            # 5. Dual update
            residual = (
                apply_affine_map(model.constraint_matrices, W)
                - model.rhs
            )
            y = y + gamma_t * residual

            # 6. Record
            cp_obj = evaluate_cp_objective(model.C, W)
            res_norm = residual_norm(residual)
            tracker.record(
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
                    f"(gap={gap:.2e}, res={res_norm:.2e})"
                )
                break

        # ---- rounding ----------------------------------------------------
        logger.info("Rounding with Section-4 rank-1 decode")
        rounding = round_from_W(
            W,
            problem,
            n_original,
            project=self.rounding_project,
        )
        best_bitstring = rounding["bitstring"]
        best_objective = rounding["objective"]
        x_hat = rounding["x_hat"]
        X_hat = rounding["X_hat"]

        # Also inspect LMO vertices (extract original-variable bits only)
        for lmo_bs in tracker.lmo_bitstrings:
            if len(lmo_bs) < 1 + n_original:
                continue
            bs = lmo_bs[1 : 1 + n_original]
            if problem.is_feasible(bs)[0]:
                cost = problem.evaluate_cost(bs)
                if cost < best_objective:
                    logger.info(
                        f"LMO vertex {bs} (cost={cost}) beats "
                        f"rounded solution (cost={best_objective})"
                    )
                    best_bitstring = bs
                    best_objective = float(cost)

        total_time = time.perf_counter() - tic_total

        classic_best = problem.get_best_solution()

        logger.info(
            f"FWALSolver: /n "
            f"Classic optimal solution: {classic_best}\n"
            f"FWAL best solution: ({best_bitstring}, {best_objective})\n"
            f"Classic walltime: {problem.wall_time:.2f}s [{problem.status}]\n"
            f"FWAL walltime: {total_time:.2f}s"
        )

        return {
            "best_bitstring": best_bitstring,
            "best_objective": best_objective,
            "continuous_solution": x_hat.tolist(),
            "continuous_objective": float(best_objective),
            "rounding_method": "rank1_projection_rounding",
            "num_iterations": len(tracker.cp_objective_values),
            "convergence": tracker.summary(),
            "total_time": total_time,
            "classic_best_bitstring": classic_best[0],
            "classic_best_objective": classic_best[1],
            "W_solution": W.tolist(),
            "X_solution": X_hat.tolist(),
            "dual_solution": y.tolist(),
            "residual_norm": (
                tracker.residual_norms[-1]
                if tracker.residual_norms
                else None
            ),
            "seed": self.seed,
        }
