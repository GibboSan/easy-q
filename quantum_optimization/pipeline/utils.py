import importlib
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Type
import yaml

from pipeline.problems.abstract_problem import AbstractProblem
from qiskit import QuantumCircuit


class OptimizationCache:

    def __init__(self, filename: str = "cache.yaml", save_every: int = 1):

        self.path = Path(filename)
        self.save_every = save_every
        self.call_counter = {}

        if self.path.exists():
            with open(self.path, "r") as f:
                self.runs = yaml.safe_load(f) or {}
        else:
            self.runs = {}

    def save(self):
        with open(self.path, "w") as f:
            yaml.safe_dump(self.runs, f, default_flow_style=False)

    def get_run(self, run_id: int) -> Optional[dict]:
        return self.runs.get(str(run_id), None)

    def is_completed(self, run_id: int) -> bool:
        run = self.get_run(run_id)
        return run is not None and run.get("completed", False)

    def start_run(self, run_id: int, init_params: list):
        run_key = str(run_id)
        if run_key not in self.runs:
            self.runs[run_key] = {
                "completed": False,
                "initial_params": init_params,
                "objective_values": [],
                "best_cost": float("inf"),
                "best_params": None,
                "nfev": 0
            }
        self.call_counter[run_key] = self.runs[run_key].get("nfev", 0)
        self.save()

    def update_run(self, run_id: int, x: list, cost: float):
        run_key = str(run_id)
        run = self.runs.get(run_key)
        if run is None:
            raise ValueError(f"Run {run_id} not initialized")

        run["objective_values"].append(cost)
        run["nfev"] = run.get("nfev", 0) + 1
        self.call_counter[run_key] = self.call_counter.get(run_key, 0) + 1

        if cost < run["best_cost"]:
            run["best_cost"] = cost
            run["best_params"] = x

        if self.call_counter[run_key] % self.save_every == 0:
            self.save()

    def complete_run(self, run_id: int):
        run = self.get_run(run_id)
        if run is not None:
            run["completed"] = True
            self.save()

    def get_all_completed(self):
        return [r for r in self.runs.values() if r.get("completed", False)]


def find_most_promising_feasible_bitstring(
        final_distribution_bin: dict,
        problem: AbstractProblem
) -> Optional[Tuple[str, float]]:

    evaluated_bitstrings = []

    for bitstring in final_distribution_bin.keys():
        if problem.is_feasible(bitstring)[0]:
            evaluated_bitstrings.append(
                (bitstring, problem.evaluate_cost(bitstring))
            )

    if not evaluated_bitstrings:
        return None

    most_promising = min(evaluated_bitstrings, key=lambda x: x[1])

    return most_promising


def hamming_distance(a, b):

    if len(a) != len(b):
        raise ValueError("Bitstrings must have the same length.")

    differences = [len(a) - i - 1 for i, (bit1, bit2) in enumerate(zip(a, b)) if bit1 != bit2]
    distance = len(differences)

    return distance, differences


def class_importer(module_name: str, class_name: str, compute_classfile_name=True) -> Type:

    if compute_classfile_name:
        classfile_name = re.sub(
            r'(?<=[a-z0-9])([A-Z])',
            r'_\1',
            re.sub(
                r'([A-Z]+)([A-Z][a-z])',
                r'\1_\2',
                class_name
            )
        ).lower()
        module_name = f"{module_name}.{classfile_name}"

    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ModuleNotFoundError:
        print(f"Error: module '{module_name}' not found.")
        sys.exit(1)
    except AttributeError:
        print(f"Error: class '{class_name}' not found in module '{module_name}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Generic error: {e}")
        sys.exit(1)


def get_circuit_metrics(qc: QuantumCircuit) -> dict:

    ops = qc.count_ops() 
    clifford_gates = {'id', 'x', 'y', 'z', 'h', 's', 'sdg', 'cx', 'cz', 'swap'}
    
    depth = qc.depth()
    num_2q = qc.num_nonlocal_gates()
    
    ignored_ops = {'barrier', 'measure', 'reset', 'snapshot'}
    total_gates = sum(count for op, count in ops.items() if op not in ignored_ops)   
    clifford_count = sum(ops.get(gate, 0) for gate in clifford_gates)
    non_clifford_count = total_gates - clifford_count

    num_active_qubits = len({q for instr in qc.data for q in instr.qubits})

    return {
        "depth": depth,
        "2q_gates": num_2q,
        "clifford_gates": clifford_count,
        "non_clifford_gates": non_clifford_count,
        "total_gates": total_gates,
        "num_active_qubits": num_active_qubits
    }


