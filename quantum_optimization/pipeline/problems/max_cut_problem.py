'''
MaxCut Problem

This module defines a random Max-Cut instance on an Erdos-Renyi graph and maps it
to a binary quadratic optimization model. Each node is assigned to one of two
partitions, and the objective prefers edges whose endpoints are in different sets.

Variables
- G=(V,E): undirected graph.
- x_i in {0,1}: partition indicator for node i (i in V).

Formal formulation
Equivalent cut maximization form:
    maximize  sum_{(u,v) in E} (x_u + x_v - 2 x_u x_v)

Implemented minimization form in this file:
    minimize  sum_{(u,v) in E} (2 x_u x_v - x_u - x_v)
    subject to x_i in {0,1} for all i in V.
'''

import networkx as nx
from qiskit_optimization import QuadraticProgram

from pipeline.problems.abstract_problem import AbstractProblem


class MaxCutProblem(AbstractProblem):

    def __init__(self, seed: int, problem_params: dict, ):
        self.n_nodes = problem_params['n_nodes']
        self.density = problem_params['density']
        super().__init__(seed, problem_params)
        
    def build_problem(self) -> QuadraticProgram:

        graph = nx.erdos_renyi_graph(self.n_nodes, self.density, seed=self.seed)
        while not nx.is_connected(graph):
            graph = nx.erdos_renyi_graph(self.n_nodes, self.density, seed=self.seed + 1)

        qp = QuadraticProgram('MAXCUT')

        for i in range(self.n_nodes):
            qp.binary_var(f"x_{i}")

        quadratic = {}
        linear = {f"x_{i}": 0 for i in range(self.n_nodes)}

        for u, v in graph.edges:
            quadratic[(f"x_{u}", f"x_{v}")] = 2
            linear[f"x_{u}"] += -1
            linear[f"x_{v}"] += -1

        qp.minimize(linear=linear, quadratic=quadratic)
        return qp