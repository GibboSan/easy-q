import itertools


from qiskit_optimization import QuadraticProgram

from pipeline.problems.abstract_problem import AbstractProblem


class ProductionPlanningProblem(AbstractProblem):

    def __init__(self, seed: int, problem_params: dict):

        self.data = {
            "small": {
                "q": {
                    ('m1', 'p1'): 1,
                    ('m1', 'p2'): 3,
                    ('m2', 'p1'): 3,
                    ('m2', 'p2'): 4
                },
                "v": {'m1': 5, 'm2': 6},
                "c": {
                    ("m1", "p1"): 2,
                    ("m1", "p2"): 1,
                    ("m2", "p1"): 1,
                    ("m2", "p2"): 3
                },
                "N": {"m1": 1, "m2": 1},
                "C": {"m1": 3, "m2": 3},
                "B": {"m1": 1, "m2": 1},
                "K": {"m1": 3, "m2": 3}
            },
            "large": {
                "q": {
                    ('m1', 'p1'): 1,
                    ('m1', 'p2'): 2,
                    ('m1', 'p3'): 3,
                    ('m1', 'p4'): 4,
                    ('m1', 'p5'): 5,
                    ('m1', 'p6'): 6,
                    ('m2', 'p1'): 3,
                    ('m2', 'p2'): 4,
                    ('m2', 'p3'): 2,
                    ('m2', 'p4'): 3,
                    ('m2', 'p5'): 4,
                    ('m2', 'p6'): 5
                },
                "v": {'m1': 5, 'm2': 6},
                "c": {
                    ("m1", "p1"): 2,
                    ("m1", "p2"): 3,
                    ("m1", "p3"): 1,
                    ("m1", "p4"): 4,
                    ("m1", "p5"): 2,
                    ("m1", "p6"): 1,
                    ("m2", "p1"): 1,
                    ("m2", "p2"): 2,
                    ("m2", "p3"): 1,
                    ("m2", "p4"): 2,
                    ("m2", "p5"): 3,
                    ("m2", "p6"): 4
                },
                "N": {"m1": 2, "m2": 3},
                "C": {"m1": 8, "m2": 7},
                "B": {"m1": 1, "m2": 1},
                "K": {"m1": 7, "m2": 7}
            },
            "medium": {
                "q": {
                    ('m1', 'p1'): 1,
                    ('m1', 'p2'): 2,
                    ('m1', 'p3'): 3,
                    ('m2', 'p1'): 3,
                    ('m2', 'p2'): 4,
                    ('m2', 'p3'): 2
                },
                "v": {'m1': 5, 'm2': 6},
                "c": {
                    ("m1", "p1"): 2,
                    ("m1", "p2"): 3,
                    ("m1", "p3"): 3,
                    ("m2", "p1"): 1,
                    ("m2", "p2"): 4,
                    ("m2", "p3"): 3
                },
                "N": {"m1": 2, "m2": 2},
                "C": {"m1": 7, "m2": 7},
                "B": {"m1": 1, "m2": 1},
                "K": {"m1": 7, "m2": 7}
            }
        }

        super().__init__(seed, problem_params)


    def build_problem(self) -> QuadraticProgram:

        size = self.problem_params["size"]

        qp = QuadraticProgram("ProductionPlanning")

        # Costs q_{m,p}
        q = self.data[size]["q"]

        # Costs v_m
        v = self.data[size]["v"]

        # Sets M and P
        M = [f"m{i + 1}" for i in range(len(v))]  # Example elements of M
        P = [f"p{i + 1}" for i in range(len(q) // len(v))]  # Example elements of P

        # Molds requirements c_{m,p}
        c = self.data[size]["c"]

        # Number of machines per group
        N = self.data[size]["N"]

        # Total molds capacity
        C = self.data[size]["C"]

        # Max number of machines operating in twin mode per group
        B = self.data[size]["B"]

        # Max unused capacity per group
        K = self.data[size]["K"]

        # Variables
        for m, p in itertools.product(M, P):
            qp.binary_var(f"x_{m}_{p}")
        # for m in M:
        #     qp.integer_var(0, B[m], f"z_{m}")
        #     qp.integer_var(0, K[m], f"y_{m}")

        # Objective function: Minimize sum(q_m_p * x_m_p) + sum(v_m * z_m)
        # qp.minimize(
        #     linear={f"x_{m}_{p}": q.get((m, p), 0) for m in M for p in P} | {f"z_{m}": v[m] for m in M}
        # )
        qp.minimize(
            linear={f"x_{m}_{p}": q.get((m, p), 0) for m in M for p in P}
        )

        # Constraints

        # Assignment constraint: sum_m x_m_p = 1 for all p
        self.constraints_sum_1 = [
            qp.linear_constraint(
                linear={f"x_{m}_{p}": 1 for m in M},
                sense="==",
                rhs=1,
                name=f"assign_{p}"
            ) for p in P
        ]

        # Num product constraint: sum_p x_m_p = N_m + z_m
        for m in M:
            # qp.linear_constraint(
            #     linear={f"x_{m}_{p}": 1 for p in P} | {f"z_{m}": -1},
            #     sense="==",
            #     rhs=N[m],
            #     name=f"num_product_{m}"
            # )
            qp.linear_constraint(
                linear={f"x_{m}_{p}": 1 for p in P},
                sense="<=",
                rhs=N[m],
                name=f"num_product_{m}"
            )

        # Num mold constraint: sum_p c_m_p * x_m_p + y_m = C_m
        # for m in M:
        #     # qp.linear_constraint(
        #     #     linear={f"x_{m}_{p}": c.get((m, p), 0) for p in P} | {f"y_{m}": 1},
        #     #     sense="==",
        #     #     rhs=C[m],
        #     #     name=f"num_mold_{m}"
        #     # )
        #     qp.linear_constraint(
        #         linear={f"x_{m}_{p}": c.get((m, p), 0) for p in P},
        #         sense="<=",
        #         rhs=C[m],
        #         name=f"num_mold_{m}"
        #     )

        return qp