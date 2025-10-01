# Quantum Optimization - EasyQ Framework

This is the **Quantum Optimization** module of the **EasyQ Framework**, a comprehensive quantum computing framework developed by [Spindox](https://makeamark.spindox.it/) within the [Ingenios](https://makeamark.spindox.it/project/ingenios/) project.

This module focuses on quantum optimization experiments with Quantum Approximate Optimization Algorithm (QAOA), providing automated experimentation and performance analysis capabilities.

> 📖 **Framework Documentation**: For installation instructions, common features, and an overview of the complete EasyQ framework (including Quantum Machine Learning), see the [main README](../README.md).

---

## 🚀 Features

- Extensible problem class with built-in functionality for classical solving, QUBO construction, and Hamiltonian computation  
- Modular QAOA circuit implementations: implement your own circuit initialization, cost operator, or mixing operator, or use the provided defaults  
- Configurable execution via parameter files  
- Backend-agnostic execution: choose among noiseless/noise-injected simulation with the [Qiskit AER simulator](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html) or execution on real quantum hardware using the [Qiskit IBM runtime](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/runtime-service)  
- Fault-tolerant design with caching and easy restart  
- Example problems (e.g., MaxCut, Quadratic Group Selection, Production Planning) and QAOA circuits for handling constraints natively  
- Automated experiment management with full SLURM support  
- Built-in plotting utilities for analysis and visualization  

---

## 🧠 Usage

### 🔹 Minimal Run

To use the basic functionality of EasyQ, define your own optimization problem using the Qiskit API by extending the abstract class `AbstractProblem`. For example, the MaxCut problem can be implemented as follows:

```python
class MaxCutProblem(AbstractProblem):

    def __init__(self, seed: int, problem_params: dict):
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
```

The already implemented methods in the abstract class handle the conversion to QUBO and the construction of the Hamiltonian operator.

Next, define the run parameters. Below is a complete list:

```yaml
seed: 12345
output_folder: "./output"
backend_name: null
is_backend_fake: True
problem_class: "MaxCutProblem"
circuit_class: "QAOACircuit"
num_layers: 3
num_starting_points: 3
lower_bound: 0.0
upper_bound: 3.15
optimization_params:
  optimizer: "SPSA"
  tolerance: 0.01
  maxiter: 30
  learning_rate: 0.8
  perturbation: 0.3
use_cache: False
cache_filename: "cache.yaml"
cache_save_every: 5
num_estimator_shots: 10_000
num_sampler_shots: 10_000
problem_params:
  n_nodes: 8
  density: 0.7
```

**Parameters:**

* `seed` – ensures reproducibility
* `output_folder` – directory for saving results
* `backend_name` – backend to use (e.g., FakeSherbrooke, ibm_kyiv)
* `is_backend_fake` – selects between AER noise-injected simulation and a real quantum backend
* `problem_class` – problem class to solve (e.g., `MaxCutProblem`)
* `circuit_class` – QAOA circuit class (e.g., `QAOACircuit`)
* `num_layers` – number of QAOA layers (p)
* `num_starting_points` – number of controlled restarts for the classical optimizer
* `lower_bound`, `upper_bound` – parameter space sampling bounds
* `optimization_params` – classical optimizer configuration
* `use_cache` – enable/disable caching
* `cache_filename` – path to cache file
* `cache_save_every` – number of evaluations before cache save
* `num_estimator_shots`, `num_sampler_shots` – shots for estimator and sampler
* `problem_params` – problem-specific parameters

Finally, execute the run by providing the parameter file:

```bash
python run.py params.yaml
```

Upon successful completion, a JSON file is generated along with plots of the quantum circuit, optimization trends, and sampling histograms. Example output:

```json
{
    "seed": 12345,
    "problem_class": "MaxCutProblem",
    "circuit_class": "QAOACircuit",
    "backend": null,
    "logic_qubits": 8,
    "physical_qubits": 8,
    "layers": 3,
    "starting_points": 3,
    "optimal_parameters": [
        2.736757728826076,
        1.3172297932489878,
        1.4069947404724406,
        0.2602236877913352,
        0.5801569541338207,
        0.6943310094277353
    ],
    "best_classic_bistring": "11010000",
    "best_classic_objective": -14.0,
    "best_classic_status": "OPTIMAL",
    "best_classic_walltime": 0.015108464,
    "best_quantum_bitstring": "10010111",
    "best_quantum_objective": -14.0,
    "best_quantum_frequency": 0.0642,
    "most_frequent_bitstring": "10010111",
    "most_frequent_objective": -14.0,
    "most_frequent_frequency": 0.0642,
    "circuit_creation_time": 0.0031547670005238615,
    "circuit_transpilation_time": 0.11524855500101694,
    "circuit_optimization_time": 29.786818634001975,
    "circuit_bounding_time": 0.0032402179967903066,
    "circuit_sampling_time": 0.09313092699812842,
    "total_quantum_walltime": 30.001593100998434
}
```

---

### 🔹 Advanced Usage

To leverage the full capabilities of EasyQ, you may override the default QAOA operators and design custom circuits tailored for specific applications (e.g., constrained optimization). Extend the `QAOACircuit` class and override the necessary methods:

```python
class CustomQAOACircuit(QAOACircuit):

    def __init__(
            self,
            seed: int,
            problem: AbstractProblem,
            num_qubits: int,
            p: int = 1,
            backend: Optional[Backend] = None
    ):
        super().__init__(seed, problem, num_qubits, p, backend)
        self.all_feasible_bitstrings = problem.all_feasible_bitstrings()

    def _initial_state(self, qreg: QuantumRegister) -> QuantumCircuit:

        valid_states_dec = [int(s, 2) for s in self.all_feasible_bitstrings]

        state_vector = np.zeros(2 ** self.num_qubits, dtype=complex)
        amplitude = 1 / np.sqrt(len(valid_states_dec))
        for idx in valid_states_dec:
            state_vector[idx] = amplitude

        qc_init = QuantumCircuit(qreg)
        qc_init.initialize(state_vector, qreg)

        return qc_init

    def _mixer_operator(self, qc: QuantumCircuit, qreg: QuantumRegister, beta: Parameter):

        valid_states_bin = self.all_feasible_bitstrings
        already_seen_pairs = set()
        already_seen_single = set()

        for i in range(len(valid_states_bin) - 1):
            for j in range(i + 1, len(valid_states_bin)):
                dist, diff = hamming_distance(valid_states_bin[i], valid_states_bin[j])

                if dist == 2:
                    q1, q2 = diff
                    if (q1, q2) not in already_seen_pairs:
                        qc.append(RXXGate(2 * beta), [qreg[q1], qreg[q2]])
                        qc.append(RYYGate(2 * beta), [qreg[q1], qreg[q2]])
                        already_seen_pairs.add((q1, q2))
                        already_seen_single.update([q1, q2])

        for i in range(self.num_qubits):
            if i not in already_seen_single:
                qc.rx(2 * beta, qreg[i])
```

Since the custom circuit inherently enforces constraints, you can also redefine the Hamiltonian computation by excluding penalties for constraint violations, as in:

```python
class QGSUncProblem(QGSProblem):

    def build_hamiltonian(self) -> SparsePauliOp:
        qubo = QuadraticProgramToQubo(penalty=0.0).convert(self._quadratic_problem)
        hamiltonian, _ = qubo.to_ising()
        return hamiltonian
```

Simply reference your custom classes in the parameter file to integrate them into your workflow.

---

### 🔹 Experiment Automation

An **experiment** is defined as a collection of input parameter files. Each file corresponds to a SLURM job and is executed as described previously. To set up an experiment, prepare the following files:

* `template.yaml` – base parameter configuration
* `variations.yaml` – defines which parameters vary and their values
* `slurm_options.yaml` – optional SLURM configuration for `sbatch` commands

**Example `variations.yaml`:**

```yaml
num_layers:
  values: [3, 5, 8, 10]
  label: "L"
num_starting_points:
  values: [5, 10]
  label: "NS"
```

This setup generates 4 × 2 = 8 configurations. Labels are used to construct descriptive filenames.

**Example `slurm_options.yaml`:**

```yaml
mem: 62G
cpus-per-task: 16
time: 48:00:00
partition: debug
```

To launch the automation process:

```bash
cd experiments_automation
python automate.py --template <template_path> --variations <variations_path> --python_exec <python_interpreter_path> --script <script_path> --slurm_options <slurm_options_path>
```

This generates a timestamped folder with subfolders for each experiment run and two scripts:

* `run_all.sh` – sequential execution of all runs
* `run_all_slurm.sh` – SLURM-based parallel execution

Example of the generated SLURM script:

```bash
#!/bin/bash
set -e
mkdir -p logs
sbatch --job-name=L3_NS5 --mem=62G --cpus-per-task=16 --time=172800 --partition=debug --output=logs/L3_NS5_%j.out --wrap="mkdir -p L3_NS5_out && python ../run.py L3_NS5/L3_NS5.yaml > L3_NS5_out/output.log"
sbatch --job-name=L3_NS10 --mem=62G --cpus-per-task=16 --time=172800 --partition=debug --output=logs/L3_NS10_%j.out --wrap="mkdir -p L3_NS10_out && python ../run.py L3_NS10/L3_NS10.yaml > L3_NS10_out/output.log"
sbatch --job-name=L5_NS5 --mem=62G --cpus-per-task=16 --time=172800 --partition=debug --output=logs/L5_NS5_%j.out --wrap="mkdir -p L5_NS5_out && python ../run.py L5_NS5/L5_NS5.yaml > L5_NS5_out/output.log"
[...]
```

---

### 🔹 Merging Outputs

To merge the results of multiple experiment runs, execute:

```bash
python merge_output.py <exp_path>
```

where `exp_path` is the root directory of the experiment. The script generates both `.csv` and `.xlsx` files aggregating all JSON outputs, enabling easy analysis and plotting.

---

## 🧠 Acknowledgments

This framework has been developed by [Spindox](https://makeamark.spindox.it/) within the scope of the [Ingenios](https://makeamark.spindox.it/project/ingenios/) project, dedicated to research and experimentation in quantum optimization.

