'''
Simple Production Planning Problem

This module defines a simplified production-planning assignment problem with
random assignment costs. Each product is assigned to exactly one machine, and each
machine has a uniform capacity limit derived from the problem dimensions.

Variables
- M: set of machines.
- P: set of products.
- q_{m,p}: random assignment cost for assigning product p to machine m.
- k: common machine capacity limit.
- x_{m,p} in {0,1}: 1 if product p is assigned to machine m, 0 otherwise.

Formal formulation
    minimize  sum_{m in M} sum_{p in P} q_{m,p} x_{m,p}
    subject to sum_{m in M} x_{m,p} = 1,  for all p in P
               sum_{p in P} x_{m,p} <= k, for all m in M
               x_{m,p} in {0,1},          for all m in M, p in P.

Problem params
- num_machines: int, number of machines in the problem instance.
- num_products: int, number of products in the problem instance.

'''

import itertools
import math

import numpy as np
from qiskit_optimization import QuadraticProgram

from pipeline.problems.abstract_problem import AbstractProblem


class SimpleProductionPlanningProblem(AbstractProblem):

    def __init__(self, seed: int, problem_params: dict):
        self.num_machines = problem_params['num_machines']
        self.num_products = problem_params['num_products']
        super().__init__(seed, problem_params)


    def build_problem(self) -> QuadraticProgram:

        qp = QuadraticProgram("SimpleProductionPlanning")

        # Costs q_{m,p}
        q = {
            (f"m{m + 1}", f"p{p + 1}"): np.random.randint(2, 10)
             for m, p in itertools.product(range(self.num_machines), range(self.num_products))
        }

        # Capacity of each machine
        k = math.ceil(1.1 * self.num_products / self.num_machines)

        # Variables
        for m, p in itertools.product(range(self.num_machines), range(self.num_products)):
            qp.binary_var(f"x_{m}_{p}")

        # Objective function: Minimize sum(q_m_p * x_m_p)
        qp.minimize(
            linear={f"x_{m}_{p}": q.get((f"m{m + 1}", f"p{p + 1}"), 0) for m in range(self.num_machines) for p in range(self.num_products)}
        )

        # Constraints

        # Assignment constraint: sum_m x_m_p = 1 for all p
        self.constraints_sum_1 = [
            qp.linear_constraint(
                linear={f"x_{m}_{p}": 1 for m in range(self.num_machines)},
                sense="==",
                rhs=1,
                name=f"assign_{p}"
            ) for p in range(self.num_products)
        ]

        # Capacity constraint: sum_p x_m_p <= k
        for m in range(self.num_machines):
            qp.linear_constraint(
                linear={f"x_{m}_{p}": 1 for p in range(self.num_products)},
                sense="<=",
                rhs=k,
                name=f"capacity_{m}"
            )

        return qp