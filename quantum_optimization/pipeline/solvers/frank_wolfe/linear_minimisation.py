"""LMO for the Q-FWAL solver.

Builds a QUBO ``AbstractProblem`` from the FWAL gradient matrix *G* and
solves it with either the classical CP-SAT solver built into
``AbstractProblem`` or the ``QAOASolver``.

Given gradient G of shape (p+1) x (p+1), the LMO minimises

    w^T G w,   w = (1, z),  z in {0,1}^p

which reduces to the unconstrained QUBO

    min_z  z^T G[1:,1:] z  +  2 G[0,1:]^T z  +  G[0,0].

Available methods:

- ``"classic"`` — CP-SAT solver from ``AbstractProblem`` (default).
- ``"qaoa"``    — ``QAOASolver`` on a quantum backend.
"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
from qiskit.providers import Backend
from qiskit_optimization import QuadraticProgram

from pipeline.problems.abstract_problem import AbstractProblem

logger = logging.getLogger("pipeline_logger")


# ------------------------------------------------------------------ #
#   QUBO problem wrapper
# ------------------------------------------------------------------ #


class _LMOQUBOProblem(AbstractProblem):
    """Wraps an FWAL gradient matrix into an ``AbstractProblem``.

    The constructor builds an unconstrained binary QUBO from *G*,
    runs the classical CP-SAT solver (inherited from ``AbstractProblem``),
    and exposes the Ising Hamiltonian for QAOA use.
    """

    def __init__(self, G: np.ndarray, seed: int):
        self._G = G
        super().__init__(seed=seed, problem_params={"p": G.shape[0] - 1})

    def build_problem(self) -> QuadraticProgram:
        G = self._G
        p = G.shape[0] - 1

        qp = QuadraticProgram("lmo_qubo")
        for i in range(p):
            qp.binary_var(f"z_{i}")

        qp.minimize(
            constant=float(G[0, 0]),
            linear=2.0 * G[0, 1:],
            quadratic=G[1:, 1:],
        )
        return qp


# ------------------------------------------------------------------ #
#   Public interface
# ------------------------------------------------------------------ #


def fwal_lmo(
    G: np.ndarray,
    seed: int,
    method: str = "classic",
    backend: Optional[Backend] = None,
    circuit_class: Optional[str] = None,
    **solver_kwargs: Any,
) -> Dict[str, Any]:
    """Solve the LMO sub-problem for the FWAL algorithm.

    Parameters
    ----------
    G : ndarray, shape (p+1, p+1)
        Symmetric FWAL gradient matrix.
    seed : int
        Random seed.
    method : str
        ``"classic"`` (default) or ``"qaoa"``.
    backend : Backend, optional
        Quantum backend; required when *method* is ``"qaoa"``.
    circuit_class : str, optional
        QAOA circuit class name; required when *method* is ``"qaoa"``.
    **solver_kwargs
        Extra keyword arguments forwarded to ``QAOASolver``.

    Returns
    -------
    dict
        ``w``             — ndarray, the lifted vertex (1, z).
        ``bitstring``     — str, ``"1" + z_string`` for vertex inspection.
        ``vertex_matrix`` — ndarray, outer product w w^T.
        ``lmo_objective`` — float, w^T G w.
        ``lmo_time``      — float, wall-clock seconds.
        ``solve_mode``    — str, ``"classic"`` or ``"qaoa"``.
    """
    tic = time.perf_counter()

    problem = _LMOQUBOProblem(G, seed)

    if method == "classic":
        best_bs, _ = problem.get_best_solution()
        solve_mode = "classic"

    elif method == "qaoa":
        from pipeline.solvers.qaoa_solver import QAOASolver

        if backend is None:
            raise ValueError("QAOA LMO requires a quantum backend.")
        if circuit_class is None:
            raise ValueError("QAOA LMO requires a circuit_class.")
        solver = QAOASolver(
            problem=problem,
            backend=backend,
            seed=seed,
            circuit_class=circuit_class,
            **solver_kwargs,
        )
        result = solver.solve(problem)
        best_bs = result["best_bitstring"]
        solve_mode = "qaoa"

    else:
        raise ValueError(f"Unknown LMO method: '{method}'")

    # Build the lifted vertex w = (1, z)
    z = np.array([int(c) for c in best_bs], dtype=float)
    w = np.concatenate(([1.0], z))
    W = np.outer(w, w)
    obj = float(w @ G @ w)

    lmo_time = time.perf_counter() - tic

    logger.debug(
        f"LMO [{solve_mode}]: obj={obj:.6f}  "
        f"z={''.join(str(int(v)) for v in z)}  time={lmo_time:.3f}s"
    )

    return {
        "w": w,
        "bitstring": "1" + best_bs,
        "vertex_matrix": W,
        "lmo_objective": obj,
        "lmo_time": lmo_time,
        "solve_mode": solve_mode,
    }
