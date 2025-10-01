from typing import Optional

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

matplotlib.use('Agg')

# def plot_graph(seed: int, graph: nx.Graph, path: str):
#     pos = nx.spring_layout(graph, seed=seed)
#     nx.draw(graph, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10, font_weight="bold")
#     plt.savefig(path)


def draw_circuit(circuit: QuantumCircuit, path: str, fold=False, idle_wires=False, scale=1.0):
    circuit_drawer(circuit, output='mpl', fold=fold, idle_wires=idle_wires, scale=scale).savefig(path)

def plot_obj_func(objective_func_vals, path):
    plt.figure()
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.grid(True)
    plt.savefig(path)
    plt.close()