# EasyQ - Quantum Calculation Made Easy

EasyQ is a flexible and extensible framework for solving both optimization and machine learning problems using quantum algorithms. The framework provides modular, extensible tools for quantum algorithm development, automated experimentation, and performance benchmarking. Built on top of [IBM Qiskit](https://www.ibm.com/quantum/qiskit) and related libraries.

This framework has been developed by [Spindox](https://makeamark.spindox.it/) within the scope of the [Ingenios](https://makeamark.spindox.it/project/ingenios/) project.

---

## 🎯 Overview

The framework is organized into two main components:

### 🔧 Quantum Optimization
Solve combinatorial optimization problems using the Quantum Approximate Optimization Algorithm (QAOA). Support for custom problem formulations, circuit designs, and constraint handling.

**Key capabilities:**
- Multiple optimization problem classes (MaxCut, TSP, scheduling, resource allocation)
- Customizable QAOA circuits with constraint-aware operators
- Classical solver integration for benchmarking
- Advanced caching and fault-tolerance mechanisms

📖 **[See Quantum Optimization Documentation →](./quantum_optimization/README.md)**

### 🧠 Quantum Machine Learning
Train quantum machine learning models using kernel methods (QSVC) and variational approaches (VQC). Comprehensive support for dataset preprocessing, dimensionality reduction, and performance analysis.

**Key capabilities:**
- Quantum Support Vector Classifiers (QSVC) with fidelity-based kernels
- Variational Quantum Classifiers (VQC) with parameterized circuits
- Built-in dataset support (benchmark, synthetic, real-world)
- Automatic classical baseline comparison

📖 **[See Quantum Machine Learning Documentation →](./quantum_machine_learning/README.md)**

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) SLURM workload manager for HPC cluster execution

### Setup

1. **Clone the repository:**
```bash
git clone <repository_url>
cd easy-q
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Core Dependencies

The framework requires the following packages (see `requirements.txt`):

**Quantum Computing:**
- `qiskit>=1.4.1,<2.0.0` - Core quantum computing library
- `qiskit-aer` - High-performance quantum simulator
- `qiskit-ibm-runtime` - IBM Quantum cloud access
- `qiskit-algorithms` - Quantum algorithms implementation
- `qiskit-optimization` - Optimization problem formulation
- `qiskit-machine-learning==0.8.3` - Quantum ML algorithms

**Scientific Computing:**
- `numpy` - Numerical computations
- `scipy` - Scientific algorithms and optimization
- `pyDOE` - Design of experiments

**Machine Learning:**
- `scikit-learn` - Classical ML algorithms and utilities
- `umap-learn` - Dimensionality reduction

**Optimization:**
- `ortools` - Google OR-Tools for classical optimization

**Utilities:**
- `matplotlib` - Visualization and plotting
- `PyYAML` - Configuration file parsing
- `networkx` - Graph operations and algorithms
- `pylatexenc` - LaTeX encoding support
- `ipython` - Interactive Python shell

### Complete Requirements File

The complete `requirements.txt` file:

```txt
matplotlib
networkx
numpy
pyDOE
pylatexenc
PyYAML
qiskit>=1.4.1,<2.0.0
qiskit-aer
qiskit-algorithms
qiskit-ibm-runtime
qiskit-optimization
qiskit-machine-learning==0.8.3
scipy
ortools
scikit-learn
umap-learn
ipython
```

## 🗂️ Project Structure

```
quantum_framework/
├── quantum_optimization/          # Optimization framework
│   ├── README.md                 # Optimization-specific documentation
│   ├── run.py                    # Main execution script
│   ├── automate.py               # Experiment automation
│   ├── pipeline/                 # Core optimization modules
│   │   ├── problems/            # Problem class implementations
│   │   ├── circuits/            # QAOA circuit implementations
│   │   └── utils.py             # Utility functions
│   └── examples/                # Example configurations
│
├── quantum_machine_learning/     # ML framework
│   ├── README.md                 # ML-specific documentation
│   ├── run.py                    # Main execution script
│   ├── automate_ML.py            # Experiment automation
│   ├── aggregate_results.py      # Results aggregation
│   ├── pipeline/                 # Core ML modules
│   │   ├── main.py              # ML pipeline orchestration
│   │   ├── methods.py           # QSVC and VQC implementations
│   │   ├── datasets.py          # Dataset utilities
│   │   ├── backends.py          # backend definition
│   │   ├── classic.py           # Classical baseline
│   │   └── plots.py             # Visualization utilities
│   └── examples/                # Example configurations
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```


## 🙏 Acknowledgments

This quantum computing framework was developed by [Spindox](https://makeamark.spindox.it/) within the scope of the [Ingenios](https://makeamark.spindox.it/project/ingenios/) project to advance research in quantum optimization and quantum machine learning.

---

## 🔗 Quick Links

- [Quantum Optimization Documentation](./quantum_optimization/README_OPT.md)
- [Quantum Machine Learning Documentation](./quantum_machine_learning/README_ML.md)
- [IBM Qiskit Documentation](https://qiskit.org/documentation/)
- [Qiskit Machine Learning](https://qiskit.org/ecosystem/machine-learning/)
- [Qiskit Optimization](https://qiskit.org/ecosystem/optimization/)
