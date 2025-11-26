from typing import Union
import numpy as np
from qiskit_optimization import QuadraticProgram

def calculate_penalty_default(qp: QuadraticProgram) -> float:
    """
    Calculates a penalty factor based on the default heuristic used in Qiskit's
    LinearEqualityToPenalty converter.
    
    It estimates the upper bound of the objective function and adds a small buffer.
    
    Args:
        qp: The QuadraticProgram instance.
        
    Returns:
        float: The calculated penalty factor.
    """
    # Get linear and quadratic coefficients of the objective
    lin = qp.objective.linear.to_array()
    quad = qp.objective.quadratic.to_array()
    
    # Calculate the maximum possible change in the objective function
    # This is a loose upper bound assuming binary variables
    max_objective = np.sum(np.abs(lin)) + np.sum(np.abs(quad))
    
    # Qiskit usually adds a small buffer or rounds up
    return max_objective + 10.0

def calculate_penalty_proportional(qp: QuadraticProgram, multiplier: float = 10.0) -> float:
    """
    Calculates a penalty factor proportional to the maximum coefficient in the objective function.
    
    Args:
        qp: The QuadraticProgram instance.
        multiplier: The factor by which to multiply the max coefficient.
        
    Returns:
        float: The calculated penalty factor.
    """
    lin = qp.objective.linear.to_array()
    quad = qp.objective.quadratic.to_array()
    
    max_coeff = 0.0
    if len(lin) > 0:
        max_coeff = max(max_coeff, np.max(np.abs(lin)))
    if len(quad) > 0:
        max_coeff = max(max_coeff, np.max(np.abs(quad)))
        
    return max_coeff * multiplier

def calculate_penalty_sum_coefficients(qp: QuadraticProgram) -> float:
    """
    Calculates a penalty factor as the sum of absolute values of all objective coefficients.
    This ensures that violating a constraint is always more expensive than any possible gain 
    from the objective function (assuming binary variables).
    
    Args:
        qp: The QuadraticProgram instance.
        
    Returns:
        float: The calculated penalty factor.
    """
    lin = qp.objective.linear.to_array()
    quad = qp.objective.quadratic.to_array()
    
    return np.sum(np.abs(lin)) + np.sum(np.abs(quad)) + 1.0

def get_penalty_calculator(method_name: str):
    """
    Factory function to retrieve a penalty calculation function by name.
    
    Args:
        method_name: The name of the method ('default', 'proportional', 'sum').
        
    Returns:
        Callable: The corresponding penalty calculation function.
    """
    methods = {
        'default': calculate_penalty_default,
        'proportional': calculate_penalty_proportional,
        'sum': calculate_penalty_sum_coefficients
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown penalty method: {method_name}. Available: {list(methods.keys())}")
        
    return methods[method_name]

"""
PROPOSED CHANGES TO abstract_problem.py:

1.  **Import the utility:**
    from pipeline.problems.penalty_utils import get_penalty_calculator

2.  **Update `__init__`:**
    Extract the penalty method and multiplier from `problem_params`.
    
    self.penalty_method = problem_params.get('penalty_method', 'default')
    self.penalty_multiplier = problem_params.get('penalty_multiplier', 10.0)

3.  **Update `build_hamiltonian`:**
    Calculate the penalty explicitly before converting to QUBO.

    def build_hamiltonian(self) -> SparsePauliOp:
        # Retrieve the calculator function
        calculator = get_penalty_calculator(self.penalty_method)
        
        # Calculate penalty
        if self.penalty_method == 'proportional':
            penalty = calculator(self._quadratic_binary_problem, self.penalty_multiplier)
        else:
            penalty = calculator(self._quadratic_binary_problem)
            
        # Use the calculated penalty in the converter
        qubo = QuadraticProgramToQubo(penalty=penalty).convert(self._quadratic_binary_problem)
        
        hamiltonian, _ = qubo.to_ising()
        return hamiltonian
"""
