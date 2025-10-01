# Quantum Machine Learning - EasyQ Framework

This is the **Quantum Machine Learning** module of the **EasyQ Framework**, a comprehensive quantum computing framework developed by [Spindox](https://makeamark.spindox.it/) within the [Ingenios](https://makeamark.spindox.it/project/ingenios/) project.

This module focuses on quantum machine learning experiments with Quantum Support Vector Classifiers (QSVC) and Variational Quantum Classifiers (VQC), providing automated experimentation and performance analysis capabilities.

> 📖 **Framework Documentation**: For installation instructions, common features, and an overview of the complete EasyQ framework (including Quantum Optimization), see the [main README](../README.md).

---

## 📋 Table of Contents

- [Supported Quantum Methods](#-Supported Quantum Methods)
- [Quantum Methods](#-quantum-methods)
- [Dataset Support](#-dataset-support)
- [Configuration](#-configuration)
- [Running Experiments](#-running-experiments)
- [Automated Experimentation](#-automated-experimentation)
- [Results Analysis](#-results-analysis)
- [Advanced Features](#-advanced-features)
- [Examples](#-examples)

---

## 🧠 Usage

### 🔹 Quick Start

Create a parameter configuration file specifying your experiment setup:

```yaml
seed: 42
output_path: "./results"
backend: "Aer"
method_type: "QSVC"
feature_map: "ZZFeatureMap"
feature_map_params:
  reps: 2
  entanglement: "full"
```

Run a single experiment:

```bash
python run.py config.yaml dataset.pkl
```

### 🔹 Supported Quantum Methods

#### Quantum Support Vector Classifier (QSVC)
Uses quantum kernel methods with fidelity-based feature mapping:

```yaml
method_type: "QSVC"
feature_map: "ZZFeatureMap"
feature_map_params:
  reps: 2
  entanglement: "full"  # Options: full, linear, circular, sca, reverse_linear
```

#### Variational Quantum Classifier (VQC)
Employs parameterized quantum circuits with classical optimization:

```yaml
method_type: "VQC"
feature_map: "ZZFeatureMap"
feature_map_params:
  reps: 2
  entanglement: "full"
ansatz: "RealAmplitudes"
ansatz_params:
  reps: 3
  entanglement: "reverse_linear"
opt_method: "COBYLA"
opt_method_params:
  maxiter: 1000
```

### Backend Selection

Choose the quantum backend for circuit execution:

```yaml
# Ideal statevector simulation (fastest, no noise)
backend: "StateVector"

# Qiskit AER simulator (shot-based, optional noise)
backend: "Aer"

# Simulate Mumbai quantum device (realistic noise model)
backend: "AerMumbai"
```

---

## 📊 Dataset Support

### Toy Datasets

Built-in sklearn datasets for quick testing:

```yaml
dataset_type: "toys"
datasets_params:
  dataset_name: ["Breast_cancer"]  # Options: Iris, Wine, Breast_cancer
```

**With Dimensionality Reduction:**

```yaml
dataset_type: "toys_red"
datasets_params:
  dataset_name: ["Breast_cancer"]
  reduction: ["pca", "umap"]       # Reduction method
  features: [2, 5, 10]             # Target dimensions
```

### Synthetic Datasets

Generate controlled datasets for systematic studies:

```yaml
dataset_type: "synthetic"
datasets_params:
  sample_size: [300, 500]          # Number of samples
  features: [2, 4, 6]              # Number of features
  class_sep: [1.0, 0.8, 0.6]       # Class separation (easier → harder)
```

**Parameters:**
- `sample_size`: Total samples (split 80/20 train/test)
- `features`: Dimensionality of feature space
- `class_sep`: Controls problem difficulty (1.0 = easy, 0.2 = hard)

### Real-World Datasets

Load and preprocess CSV datasets:

```yaml
dataset_type: "real"
file_path: "path/to/dataset.csv"
datasets_params:
  dataset_name: ["unsw_nb15"]
  sample_size: [500, 1000]
  reduction: ["pca", "umap"]
  features: [2, 5]
  sampling: ["balanced"]           # Balance class distribution
```

**Preprocessing Pipeline:**
1. Load CSV file
2. Encode categorical variables
3. Apply dimensionality reduction (if specified)
4. Scale features (StandardScaler)
5. Split train/test (80/20, stratified)
6. Optional balanced sampling for imbalanced datasets

---

## ⚙️ Configuration

### Complete Configuration Example

```yaml
seed: 42
output_path: "./results"
backend: "Aer"
method_type: "QSVC"
feature_map: "ZZFeatureMap"
feature_map_params:
  reps: 2
  entanglement: "full"
```

### Configuration Parameters

**Global Settings:**
- `seed`: Random seed for reproducibility
- `output_path`: Directory for results and models
- `backend`: Quantum backend selection

**Method Configuration:**
- `method_type`: "QSVC" or "VQC"
- `feature_map`: Feature map circuit type
- `feature_map_params`: Feature map parameters

**VQC-Specific:**
- `ansatz`: Ansatz circuit type
- `ansatz_params`: Ansatz parameters
- `opt_method`: Classical optimizer
- `opt_method_params`: Optimizer parameters

**Entanglement Options:**
- `full`: All-to-all qubit connectivity (maximum expressiveness)
- `linear`: Chain connectivity (hardware-efficient)
- `circular`: Ring connectivity
- `sca`: Shifted circular alternating
- `reverse_linear`: Reverse chain connectivity

**Key Parameters:**
- `reps`: Number of feature map repetitions (controls expressiveness vs. depth)
- Higher `reps` → More expressive but deeper circuits
- Typical range: 1-3 for most applications

**Supported Optimizers:**
- `COBYLA`: Gradient-free, robust, good for noisy objectives
- `SLSQP`: Sequential Least Squares Programming
- `L_BFGS_B`: Limited-memory BFGS with bounds

**Ansatz Design:**
- `RealAmplitudes`: Rotation + entanglement layers
- Custom ansätze can be added to `CIRCUIT_TYPES` dictionary

---

## 🚀 Running Experiments

### Single Experiment

Execute a single configuration:

```bash
python run.py config.yaml dataset.pkl
```

**Arguments:**
- `config.yaml`: Parameter configuration file
- `dataset.pkl`: Preprocessed dataset (pickle format)

### Output Files

Each experiment generates:

```
output_folder/
├── output.json              # Comprehensive results
├── output.log               # Execution log
├── classifier.model         # Trained quantum model (VQC)
├── kernel_train.npy         # Training kernel matrix (QSVC)
├── kernel_test.npy          # Test kernel matrix (QSVC)
├── objective_func.csv       # Optimization history (VQC)
└── objective_function.png   # Convergence plot (VQC)
```

### Results Structure

`output.json` contains:

```json
{ "output_path": "..\\ML_exp_xxDATExx_xxTIMExx_QSVC\\synt_50_2_0.2\\1_reps_map\\linear_entangl_map",
  "seed": 42,
  "backend": "Aer",
  "method_type": "QSVC",
  "feature_map": "ZZFeatureMap",
    "feature_map_params": {
        "reps": 1,
        "entanglement": "linear"
    },
  "n_qubit_feat_map": 2,
  "circuit_depth_feat_map": 5,
  "circuit_n_gate_feat_map": 7,
  "circuit_n_cnot_feat_map": 2,
  "n_qubit_ansatz": null,
  "circuit_depth_ansatz": null,
  "circuit_n_gate_ansatz": null,
  "circuit_n_cnot_ansatz": null,
  "quantum_results": {
    "train_score": 0.95,
    "test_score": 0.92,
    "walltime": 45.2,
    "walltime_kernel_train": 35.1,
    "walltime_kernel_test": 8.7
  },
  "classic_SVC": {
    "train_score": 0.93,
    "test_score": 0.91,
    "walltime": 0.15
  }
}
```

---

## 🔄 Automated Experimentation

### Configuration Files

Create configuration files for automated experiments:

**1. Variations Configuration** (`variations_config.yaml`):

```yaml
seed: 42
output_folder_path: "./experiments"
backend: "Aer"
dataset_type: "synthetic"
datasets_params:
  sample_size: [200, 300]
  features: [2, 3, 4]
  class_sep: [1.0, 0.8]
quantum_methods: "QSVC"
methods_params:
  reps_map: [1, 2, 3]
  entangl_map: ["full", "linear"]
```

**2. SLURM - Optional** (`slurm_options.yaml`):

```yaml
mem: 28G
cpus-per-task: 8
time: 36:00:00
partition: debug
```

### Launch Automated Experiments

Generate experiment structure:

```bash
python automate_ML.py \
  --configuration_runs variations_config.yaml \
  --python_exec python \
  --script run.py \
  --slurm_options slurm_options.yaml
```

**Generated Structure:**

```
exp_20250930_182652_QSVC/
├── synt_50_2_0.2/
│   ├── 1_reps_map/linear_entangl_map/
│   │   ├── experimental_parameter.yaml
│   │   ├── kernel_train.npy
│   │   ├── kernel_test.npy
│   │   ├── output.json
│   │   └── output.log
│   ├── 2_reps_map/full_entangl_map/
│   └── ...
├── synt_50_2_0.4/
├── run_all.sh              # Sequential execution
└── run_all_slurm.sh        # Parallel SLURM execution
```

### Execute Experiments

**Sequential:**
```bash
cd exp_20250930_182652_QSVC
./run_all.sh
```

**Parallel (SLURM):**
```bash
cd exp_20250930_182652_QSVC
./run_all_slurm.sh
```

---

## 📊 Results Analysis

### Aggregate Results

Collect all experiment results into structured dataframes:

```bash
python aggregate_results.py <experiment_folder> <output_folder>
```

**Example:**
```bash
python aggregate_results.py \
  experiments/exp_20250930_182652_QSVC \
  reports/qsvc_analysis
```

### Output Formats

The script generates a CSV file with the name of the experimental_folder:

- **CSV** (`<experiment_folder>.csv`): ex. exp_20250930_182652_QSVC.csv

---

## 🔧 Advanced Features

### Custom Circuit Implementation

Extend the framework with custom quantum circuits:

```python
from qiskit.circuit.library import NLocal
from pipeline.methods import CIRCUIT_TYPES

def custom_feature_map(num_features, **params):
    """Custom feature map implementation"""
    return NLocal(
        num_features,
        rotation_blocks=['ry', 'rz'],
        entanglement_blocks='cz',
        entanglement=params.get('entanglement', 'full'),
        reps=params.get('reps', 1)
    )

# Register custom circuit
CIRCUIT_TYPES["CustomFeatureMap"] = {
    "function": custom_feature_map
}
```

### Parallel Kernel Computation

The QSVC implementation includes optimized kernel computation:

**Features:**
- Row-by-row computation with symmetry exploitation
- Progress logging every 25 rows
- Memory-efficient for large kernel matrices
- Automatic result caching

**Implementation details:**
```python
# Symmetric kernel computation
K_train = compute_kernel_block(train_features, kernel)

# Asymmetric kernel computation
K_test = compute_kernel_block(test_features, kernel, train_features)
```

---

## 📚 Examples

The repository includes several example configurations:

- `variations_toys.yaml` - Experiments on sklearn toy datasets
- `variations_synt.yaml` - Synthetic data experiments with complexity analysis
- `variations_real.yaml` - Real-world dataset experiments with preprocessing
- `variations_synt_vqc.yaml` - VQC-specific parameter exploration

---

## 📖 Further Reading

- **Main Framework Documentation**: [../README.md](../README.md)
- **Qiskit Machine Learning**: [Documentation](https://qiskit.org/ecosystem/machine-learning/)
- **Classification Methods**: [Tutorial](https://qiskit.org/ecosystem/machine-learning/tutorials/)

---

For general installation, common features, and framework overview, refer to the [main README](../README.md).

---

## 🧠 Acknowledgments

EasyQ framework was developed by [Spindox](https://makeamark.spindox.it/) within the scope of the [Ingenios](https://makeamark.spindox.it/project/ingenios/) project to advance research in quantum-enhanced machine learning and enable systematic comparison of quantum vs classical approaches.
