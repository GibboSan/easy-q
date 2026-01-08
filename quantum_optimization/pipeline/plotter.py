import os
from itertools import islice

import matplotlib.pyplot as plt
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

from pipeline.problems.abstract_problem import AbstractProblem


class Plotter:

    def __init__(self, folder: str):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)

    def draw_circuit(
            self, qc: QuantumCircuit,
            filename: str,
            fold=False,
            idle_wires=False,
            scale=1.0
    ) -> None:

        if not filename.endswith(".png"):
            filename += ".png"

        filepath = os.path.join(self.folder, filename)
        fig = circuit_drawer(qc, output='mpl', fold=fold, idle_wires=idle_wires, scale=scale)
        fig.savefig(filepath)

    def plot_parameter_optimization(
            self,
            objective_fun_vals_all: list[list[float]],
            final_params_all: list[list[float]],
            filename: str
    ) -> None:

        if not filename.endswith(".png"):
            filename += ".png"
        filepath = os.path.join(self.folder, filename)

        plt.figure()

        num_plot = len(objective_fun_vals_all)

        for run in range(num_plot):
            plt.plot(
                range(len(objective_fun_vals_all[run])),
                objective_fun_vals_all[run],
                label=f"Run {run + 1}: {final_params_all[run]}"
            )

        plt.xlabel('Iteration Number')
        plt.ylabel('Objective Function Value')
        plt.title('Objective Function Values Across Iterations for Each Run')
        plt.grid(True)
        plt.savefig(filepath)

    def plot_bitstring_distribution(
            self,
            final_distribution_bin: dict,
            problem: AbstractProblem,
            filename: str,
            n: int = 100
    ) -> None:

        if not filename.endswith(".png"):
            filename += ".png"
        filepath = os.path.join(self.folder, filename)

        if len(final_distribution_bin) > n:
            final_distribution_bin = dict(islice(final_distribution_bin.items(), n))

        bitstring_list, probability_list = zip(*final_distribution_bin.items())

        feas_list = [problem.is_feasible(bs)[0] for bs in bitstring_list]

        colors = ['green' if feasible else 'red' for feasible in feas_list]
        obj_values = [problem.evaluate_cost(bitstring_list[i]) if feas_list[i] else None for i in range(len(bitstring_list))]

        plt.figure(figsize=(20, 6))
        bars = plt.bar(bitstring_list, probability_list, color=colors)

        for bar, obj in zip(bars, obj_values):
            if obj is not None:
                height = 1.1 * bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{obj:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=6,
                    color='black',
                    rotation=90
                )

        plt.title("Result Distribution")
        plt.xlabel('Bitstring')
        plt.ylabel('Probability')
        plt.xticks(rotation=90, fontsize=6)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(filepath)

    def generate_frequency_report(
            self,
            distribution: dict,
            problem: AbstractProblem,
            filename: str
    ):

        if not filename.endswith(".csv"):
            filename += ".csv"
        filepath = os.path.join(self.folder, filename)

        constraints_name_list = [constraint.name for constraint in problem.quadratic_binary_problem.linear_constraints]

        column_names_list = ["index", "bitstring", "frequency", "objective", "is_feasible"] + constraints_name_list

        row_dict = {
            column_name: []
            for column_name in column_names_list
        }

        for index, (bitstring, frequency) in enumerate(distribution.items()):

            row_dict["index"].append(index)
            row_dict["bitstring"].append(bitstring)
            row_dict["frequency"].append(frequency)
            row_dict["objective"].append(problem.evaluate_cost(bitstring))

            is_feasible, constraint_satisfaction_dict = problem.is_feasible(bitstring, True)

            row_dict["is_feasible"].append(is_feasible)

            for constraint_name, satisfied in constraint_satisfaction_dict.items():
                row_dict[constraint_name].append(satisfied)

        df = pd.DataFrame(row_dict)

        df.to_csv(filepath, index=False)

