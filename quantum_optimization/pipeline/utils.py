import importlib
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Type
import yaml
import math

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

    clifford_gates_names = {
        'id', 'x', 'y', 'z', 'h', 's', 'sdg', 'cx', 'cz', 'swap', 'sx', 'ecr', 'iswap'}
    
    ignored_ops = {'barrier', 'measure', 'reset', 'snapshot', 'delay', 'initialize'}
    
    depth = qc.depth()
    num_2q = qc.num_nonlocal_gates()
    
    clifford_count = 0
    non_clifford_count = 0
    t_count = 0 
    
    total_gates = 0
    
    pi_half = math.pi / 2
    pi_quarter = math.pi / 4 

    for instr in qc.data:
        op = instr.operation
        name = op.name
        
        if name in ignored_ops:
            continue
            
        total_gates += 1
        
        if name in clifford_gates_names:
            clifford_count += 1
            continue
            
        is_clifford = False
        
        if name in {'rz', 'p', 'rx', 'ry'}:
            if len(op.params) > 0 and isinstance(op.params[0], (int, float)):
                try:
                    angle = float(op.params[0])
                    is_t = math.isclose(angle, pi_quarter, abs_tol=1e-9)
                    is_tdg = math.isclose(angle, -pi_quarter, abs_tol=1e-9)

                    if is_t or is_tdg:
                        t_count += 1
                        non_clifford_count += 1
                        continue 

                    steps = angle / pi_half
                    if math.isclose(steps, round(steps), abs_tol=1e-9):
                        clifford_count += 1
                        is_clifford = True
                        
                except Exception:
                    pass
        
        elif name == 't' or name == 'tdg':
             t_count += 1
             non_clifford_count += 1
             continue
             
        elif name in {'u', 'u1', 'u2', 'u3', 'rxx', 'rzz', 'ryy'}:
            pass

        if not is_clifford:
            non_clifford_count += 1

    num_active_qubits = len({q for instr in qc.data for q in instr.qubits})
    
    return {
        "depth": depth,
        "2q_gates": num_2q,
        "clifford_gates": clifford_count,
        "non_clifford_gates": non_clifford_count,
        "t_gates": t_count,
        "total_gates": total_gates,
        "num_active_qubits": num_active_qubits
    }


def compute_approximation_ratio(energy_found: float, energy_optimal: float) -> float:

    if abs(energy_optimal) < 1e-9:
        return 1.0 if abs(energy_found) < 1e-9 else 0.0
        
    return energy_found / energy_optimal


def compute_hellinger_distance(p: dict, q: dict) -> float:

    all_keys = set(p.keys()) | set(q.keys())
    
    sum_sq_diff = 0.0
    for key in all_keys:
        p_val = p.get(key, 0.0)
        q_val = q.get(key, 0.0)
        sum_sq_diff += (math.sqrt(p_val) - math.sqrt(q_val)) ** 2
        
    return (1.0 / math.sqrt(2.0)) * math.sqrt(sum_sq_diff)


def compute_js_divergence(p: dict, q: dict) -> float:
    """
    Computes the Jensen-Shannon Divergence between two probability distributions.
    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)
    
    Uses log base 2, so the result is bounded between 0 and 1.
    """
    all_keys = set(p.keys()) | set(q.keys())
    
    m = {}
    for key in all_keys:
        p_val = p.get(key, 0.0)
        q_val = q.get(key, 0.0)
        m[key] = 0.5 * (p_val + q_val)
        
    def kl_divergence(dist_a, dist_b):
        kl = 0.0
        for key, val_a in dist_a.items():
            if val_a > 0:
                val_b = dist_b.get(key, 0.0)
                if val_b > 0:
                    kl += val_a * math.log2(val_a / val_b)
        return kl

    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def analyze_distribution(
    distribution: dict, 
    problem: AbstractProblem, 
    optimal_cost: Optional[float] = None
) -> dict:

    avg_energy = 0.0
    success_probability = 0.0
    best_feasible_bitstring = None
    best_feasible_cost = float('inf')
    best_feasible_frequency = None
    
    tol = 1e-5

    for bitstring, probability in distribution.items():

        cost = problem.evaluate_cost(bitstring)
        avg_energy += cost * probability

        is_feasible, _ = problem.is_feasible(bitstring)     
        if is_feasible:
            if cost < best_feasible_cost:
                best_feasible_cost = cost
                best_feasible_bitstring = bitstring
            
            if optimal_cost is not None and abs(cost - optimal_cost) < tol:
                success_probability += probability
    
    best_feasible_frequency = distribution[best_feasible_bitstring] if best_feasible_bitstring else None
    
    most_frequent = list(distribution.items())[0]
    most_frequent_bitstring = most_frequent[0]
    most_frequent_cost = problem.evaluate_cost(most_frequent_bitstring)
    most_frequent_frequency = most_frequent[1]

    best_feasible = (best_feasible_bitstring, best_feasible_cost, best_feasible_frequency)
    most_frequent = (most_frequent_bitstring, most_frequent_cost, most_frequent_frequency)

    return best_feasible, most_frequent, avg_energy, success_probability



