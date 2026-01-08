import numpy as np
from qiskit_optimization import QuadraticProgram

from pipeline.problems.abstract_problem import AbstractProblem


class QGSProblem(AbstractProblem):


    def __init__(self, seed: int, problem_params: dict):
        self.num_groups = problem_params['num_groups']
        self.instruments_per_group = problem_params['num_instruments_per_group']
        super().__init__(seed, problem_params)


    def build_problem(self) -> QuadraticProgram:

        total_vars = self.num_groups * self.instruments_per_group

        # Generate variable names
        variables = [f"x{i}" for i in range(total_vars)]

        # Build groups automatically
        groups = {
            f"G{g + 1}": variables[g * self.instruments_per_group:(g + 1) * self.instruments_per_group]
            for g in range(self.num_groups)
        }

        # Create a random symmetric correlation matrix
        corr = np.random.uniform(-1, 1, size=(total_vars, total_vars))
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 0)

        qp = QuadraticProgram("Correlation_Based_Selection")

        # Add binary variables
        for var in variables:
            qp.binary_var(var)

        # Add one-per-group constraints
        self.constraints_sum_1 = [
            qp.linear_constraint(
                linear={var: 1 for var in group_vars},
                sense="==",
                rhs=1,
                name=f"{group_name}_select_one"
            ) for group_name, group_vars in groups.items()
        ]

        # Build the quadratic objective: sum over i < j
        quadratic = {}
        for i in range(total_vars):
            for j in range(i + 1, total_vars):
                key = (variables[i], variables[j])
                quadratic[key] = corr[i, j]

        # Minimize total correlation among selected instruments
        qp.minimize(quadratic=quadratic)

        return qp
