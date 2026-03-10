"""CPLEX-based LMO using docplex to solve the unconstrained QUBO.

Requires the ``docplex`` package (and optionally a full CPLEX licence
for large instances).  The community edition bundled with docplex handles
moderate-size QUBOs.
"""

import logging
import time
from typing import Dict

import numpy as np
from docplex.mp.model import Model

from pipeline.solvers.frank_wolfe.lmo.abstract_lmo import AbstractLMO

logger = logging.getLogger("pipeline_logger")


class CplexLMO(AbstractLMO):
    """LMO that solves  min w^T G w  via CPLEX (docplex).

    Parameters
    ----------
    time_limit : float
        Maximum solve time in seconds (default 60).
    log_output : bool
        Whether to print CPLEX solver logs (default False).
    """

    def __init__(self, time_limit: float = 60.0, log_output: bool = False):
        self.time_limit = time_limit
        self.log_output = log_output

    def solve(self, G: np.ndarray, seed: int) -> Dict:
        tic = time.perf_counter()
        G_sym = (G + G.T) / 2.0
        p = G_sym.shape[0]

        model = Model(name="lmo_qubo")
        model.parameters.timelimit = self.time_limit
        model.parameters.randomseed = seed % (2**31)
        if not self.log_output:
            model.set_log_output(None)

        w_vars = [model.binary_var(name=f"w_{i}") for i in range(p)]

        # Build quadratic objective  w^T G w
        obj = model.sum(
            G_sym[i, j] * w_vars[i] * w_vars[j]
            for i in range(p)
            for j in range(p)
            if abs(G_sym[i, j]) > 1e-12
        )
        model.minimize(obj)

        solution = model.solve()

        if solution is None:
            logger.warning("CplexLMO: no solution found, returning zero vector")
            w = np.zeros(p, dtype=float)
            lmo_val = 0.0
        else:
            w = np.array(
                [round(solution.get_value(w_vars[i])) for i in range(p)],
                dtype=float,
            )
            lmo_val = float(w @ G_sym @ w)

        H = np.outer(w, w)
        return {
            "w": w,
            "bitstring": "".join(str(int(v)) for v in w),
            "vertex_matrix": H,
            "lmo_objective": lmo_val,
            "lmo_time": time.perf_counter() - tic,
            "solve_mode": "cplex",
        }
