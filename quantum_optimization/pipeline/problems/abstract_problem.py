from abc import ABC, abstractmethod
from typing import List, Tuple

from ortools.sat.python import cp_model
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo, IntegerToBinary
from qiskit_optimization.problems import LinearConstraint


class AbstractProblem(ABC):

    def __init__(self, seed: int, problem_params: dict):

        self.constraints_sum_1 = None
        self.seed = seed
        self.problem_params = problem_params
        self._quadratic_problem = self.build_problem()

        if not isinstance(self._quadratic_problem, QuadraticProgram):
            raise ValueError("build_problem must return a QuadraticProgram instance")

        self._quadratic_binary_problem = IntegerToBinary().convert(self._quadratic_problem)
        self._hamiltonian = self.build_hamiltonian()
        self._best_solution, self._best_cost, self.wall_time, self.status = self._solve()


    @abstractmethod
    def build_problem(self) -> QuadraticProgram:
        pass

    @property
    def quadratic_binary_problem(self):
        return self._quadratic_binary_problem

    @property
    def quadratic_problem(self) -> QuadraticProgram:
        return self._quadratic_problem

    @property
    def hamiltonian(self):
        return self._hamiltonian

    def build_hamiltonian(self) -> SparsePauliOp:
        qubo = QuadraticProgramToQubo().convert(self._quadratic_binary_problem)
        hamiltonian, _ = qubo.to_ising()
        return hamiltonian

    def _solve(self) -> Tuple[str, float, float, str]:

        qubo = QuadraticProgramToQubo().convert(self.quadratic_binary_problem)

        quadratic_matrix = qubo.objective.quadratic.to_array()
        linear_coeffs = qubo.objective.linear.to_array()

        num_vars = len(linear_coeffs)

        model = cp_model.CpModel()

        x = [model.NewBoolVar(f'x_{i}') for i in range(num_vars)]

        scale_factor = 1000

        linear_terms = []
        for i in range(num_vars):
            if abs(linear_coeffs[i]) > 1e-10:
                coeff = int(linear_coeffs[i] * scale_factor)
                linear_terms.append(coeff * x[i])

        quadratic_terms = []
        for i in range(num_vars):
            for j in range(i, num_vars):
                if abs(quadratic_matrix[i, j]) > 1e-10:
                    coeff = int(quadratic_matrix[i, j] * scale_factor)
                    if i == j:
                        quadratic_terms.append(coeff * x[i])
                    else:
                        product_var = model.NewBoolVar(f'prod_{i}_{j}')

                        model.Add(product_var <= x[i])
                        model.Add(product_var <= x[j])
                        model.Add(product_var >= x[i] + x[j] - 1)

                        total_coeff = coeff
                        if i != j and abs(quadratic_matrix[j, i]) > 1e-10:
                            total_coeff += int(quadratic_matrix[j, i] * scale_factor)

                        quadratic_terms.append(total_coeff * product_var)

        objective_terms = linear_terms + quadratic_terms
        if objective_terms:
            model.Minimize(sum(objective_terms))

        solver = cp_model.CpSolver()

        solver.parameters.max_time_in_seconds = 300.0
        solver.parameters.log_search_progress = False

        status = solver.Solve(model)

        status_dict = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.UNKNOWN: "UNKNOWN"
        }

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution_binary = ''.join([str(solver.Value(x[i])) for i in range(num_vars)])
            return solution_binary, self.evaluate_cost(solution_binary), solver.wall_time, status_dict[status]

        raise RuntimeError(f"CPSAT solver cannot find a feasible solution. Status: {status}")

    def get_best_solution(self) -> Tuple[str, float]:
        return self._best_solution, self._best_cost

    def evaluate_cost(self, solution: str) -> float:

        variable_values = {var.name: int(solution[i]) for i, var in enumerate(self.quadratic_binary_problem.variables)}

        linear_coeffs = {
            self.quadratic_binary_problem.variables[i].name: coeff for i, coeff in self.quadratic_binary_problem.objective.linear.to_dict().items()
        }

        quadratic_coeffs = {
            (self.quadratic_binary_problem.variables[i].name, self.quadratic_binary_problem.variables[j].name): coeff
            for (i, j), coeff in self.quadratic_binary_problem.objective.quadratic.to_dict().items()
        }

        constant_term = self.quadratic_binary_problem.objective.constant

        linear_contribution = sum(
            variable_values[var] * coeff
            for var, coeff in linear_coeffs.items()
        )

        quadratic_contribution = sum(
            variable_values[var1] * variable_values[var2] * coeff
            for (var1, var2), coeff in quadratic_coeffs.items()
        )

        return float(linear_contribution + quadratic_contribution + constant_term)

    def is_feasible_old(self, solution: str, verbose: bool = False) -> Tuple[bool, dict]:
        
        variable_values = {var.name: int(solution[i]) for i, var in enumerate(self.quadratic_binary_problem.variables)}

        constraint_violation_dict = {}

        for constraint in self.quadratic_binary_problem.linear_constraints:

            coeffs = {self.quadratic_binary_problem.variables[i].name: coeff for i, coeff in constraint.linear.to_dict().items()}

            lhs_value = sum(variable_values[var] * coeff for var, coeff in coeffs.items())

            if constraint.sense == constraint.Sense.EQ:

                condition_for_violation = (lhs_value != constraint.rhs)

                if verbose:
                    constraint_violation_dict[constraint.name] = condition_for_violation

                if condition_for_violation:
                    if not verbose:
                        return False, constraint_violation_dict

            if constraint.sense == constraint.Sense.GE:

                condition_for_violation = (lhs_value < constraint.rhs)

                if verbose:
                    constraint_violation_dict[constraint.name] = condition_for_violation

                if condition_for_violation:
                    if not verbose:
                        return False, constraint_violation_dict

            if constraint.sense == constraint.Sense.LE:

                condition_for_violation = (lhs_value > constraint.rhs)

                if verbose:
                    constraint_violation_dict[constraint.name] = condition_for_violation

                if condition_for_violation:
                    if not verbose:
                        return False, constraint_violation_dict

        if not verbose:
            return True, constraint_violation_dict

        return all(constraint_violation_dict.values()), constraint_violation_dict

    def is_feasible(self, solution: str, verbose: bool = False) -> Tuple[bool, dict]:

        ops = {
            LinearConstraint.Sense.EQ: lambda lhs, rhs: lhs == rhs,
            LinearConstraint.Sense.GE: lambda lhs, rhs: lhs >= rhs,
            LinearConstraint.Sense.LE: lambda lhs, rhs: lhs <= rhs,
        }

        variable_values = {
            var.name: int(solution[i])
            for i, var in enumerate(self.quadratic_binary_problem.variables)
        }

        constraint_satisfaction_dict = {}

        for constraint in self.quadratic_binary_problem.linear_constraints:

            coeffs = {
                self.quadratic_binary_problem.variables[i].name: coeff
                for i, coeff in constraint.linear.to_dict().items()
            }

            lhs_value = sum(variable_values[var] * coeff for var, coeff in coeffs.items())
            is_satisfied = ops[constraint.sense](lhs_value, constraint.rhs)

            if verbose:
                constraint_satisfaction_dict[constraint.name] = bool(is_satisfied)
            if not is_satisfied and not verbose:
                return False, constraint_satisfaction_dict

        if not verbose:
            return True, {}

        return all(constraint_satisfaction_dict.values()), constraint_satisfaction_dict

    def all_feasible_bitstrings(self) -> List[str]:
        n_qubit = self.hamiltonian.num_qubits
        return [
            format(i, f"0{n_qubit}b")
            for i in range(2 ** n_qubit)
            if self.is_feasible(format(i, f"0{n_qubit}b"))[0]
        ]

    def get_qubit_subsets_from_sum1_constraints(self) -> List[List[int]]:
        subsets = []
        for constraint in self.constraints_sum_1:
            subsets.append(list(constraint.linear.to_dict().keys()))
        return subsets

    def get_feasible_logic_expression_and_total_from_sum1_constraints(self) -> Tuple[str, int]:

        subsets = self.get_qubit_subsets_from_sum1_constraints()

        if not subsets:
            return "", 0

        total = 1
        xor_list = []
        for subset in subsets:
            total *= len(subset)
            xor_list.append(f"({' ^ '.join([f'x_{q}' for q in subset])})")

        return " & ".join(xor_list), total
