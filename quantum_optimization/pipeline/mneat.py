"""MNEAT: Mitigated Noise Estimator Analyzer Tool.

This module extends Qiskit Runtime's NEAT class with Mitiq-based mitigation
simulation methods that preserve the NEAT return type (NeatResult).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import numpy as np

from qiskit import QuantumCircuit, circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives.containers import EstimatorPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.debug_tools import Neat
from qiskit_ibm_runtime.debug_tools.neat_results import NeatPubResult, NeatResult

try:
    from qiskit_aer.primitives.estimator_v2 import EstimatorV2 as AerEstimator
    from qiskit_aer.primitives.sampler_v2 import SamplerV2 as AerSampler
except ImportError as exc:
    raise ImportError(
        "quantum_optimization/mneat.py requires qiskit-aer."
    ) from exc

try:
    from mitiq import zne
    from mitiq.ddd import execute_with_ddd
    from mitiq.cdr import execute_with_cdr
    from mitiq.pec import (
        execute_with_pec,
        represent_operations_in_circuit_with_local_depolarizing_noise as represent_ops_local,
    )
    try:
        # Docs: mitiq.experimental.pea.pea.execute_with_pea
        from mitiq.experimental.pea.pea import execute_with_pea
    except Exception:  # pragma: no cover
        # Fallback for alternate export styles.
        from mitiq.experimental import pea as _pea_module

        execute_with_pea = _pea_module.execute_with_pea
    from mitiq.rem import (
        execute_with_rem,
        generate_tensored_inverse_confusion_matrix
    )
    from mitiq.zne.inference import RichardsonFactory

    from mitiq import Observable as MitiqObservable
    from mitiq import PauliString as MitiqPauliString

    HAS_MITIQ = True
except ImportError:
    HAS_MITIQ = False

MitigationTechnique = Literal["zne", "pec", "rem", "cdr", "ddd", "pea"]
ExecutionMode = Literal["estimator", "sampler"]


class MNeat(Neat):
    """NEAT extension with Mitiq-backed mitigated noisy simulations.

    Core API:
      - mitigated_sim(..., technique=..., technique_kwargs=...)
    """

    def _ensure_mitiq(self) -> None:
        if not HAS_MITIQ:
            raise ValueError(
                "Mitiq is required for mitigated simulations. Install mitiq and retry."
            )

    @staticmethod
    def _as_observable_array(observables: Any) -> np.ndarray:
        """Normalize PUB observables to an object array preserving shape."""
        try:
            obs_array = np.asarray(observables, dtype=object)
            if obs_array.shape == ():
                obs_array = obs_array.reshape(1)
            return obs_array
        except Exception:
            return np.asarray([observables], dtype=object)

    @staticmethod
    def _extract_scalar(value: Any) -> float:
        """Coerce mitigation outputs (float, ndarray, tuple) to a float."""
        if isinstance(value, tuple):
            value = value[0]
        arr = np.asarray(value).reshape(-1)
        if arr.size != 1:
            raise ValueError(
                "Mitigation executor returned a non-scalar value. Use scalar PUB expectations."
            )

        scalar = arr[0]
        if np.iscomplexobj(scalar):
            imag_part = float(np.imag(scalar))
            if not np.isclose(imag_part, 0.0, atol=1e-10):
                raise ValueError(
                    "Mitigation executor returned a complex scalar with a non-negligible imaginary part."
                )
            return float(np.real(scalar))

        return float(scalar)

    @staticmethod
    def _extract_counts_from_sampler_data(data: Any) -> dict[str, int]:
        """Extract count dictionary from Qiskit Sampler result data."""
        classic_register = getattr(data, "classic_register", None)
        if classic_register is not None and hasattr(classic_register, "get_counts"):
            return classic_register.get_counts()

        join_data = getattr(data, "join_data", None)
        if callable(join_data):
            joined = join_data()
            get_counts = getattr(joined, "get_counts", None)
            if callable(get_counts):
                return cast(dict[str, int], get_counts())

        for attr in dir(data):
            if attr.startswith("_"):
                continue
            value = getattr(data, attr, None)
            get_counts = getattr(value, "get_counts", None)
            if callable(get_counts):
                return cast(dict[str, int], get_counts())

        raise ValueError("Unable to extract counts from Sampler result data.")

    @staticmethod
    def _active_qubit_indices(circuit: Any) -> list[int]:
        """Return sorted active qubit indices in a Qiskit circuit."""
        if circuit is None:
            return []
        active: set[int] = set()
        for instruction in getattr(circuit, "data", []):
            for qubit in getattr(instruction, "qubits", []):
                try:
                    active.add(circuit.find_bit(qubit).index)
                except Exception:
                    continue
        return sorted(active)

    @staticmethod
    def _project_pauli_label(label: str, circuit: Any) -> str:
        """Project a full-width Pauli label onto the circuit active qubits."""
        if circuit is None:
            return label

        active_indices = MNeat._active_qubit_indices(circuit)
        if not active_indices:
            return label
        if len(label) == len(active_indices):
            return label

        local_label_fwd = ["I"] * len(active_indices)
        local_label_rev = ["I"] * len(active_indices)
        n = len(label)
        for local_idx, global_idx in enumerate(active_indices):
            if 0 <= global_idx < len(label):
                local_label_fwd[local_idx] = label[global_idx]
                local_label_rev[local_idx] = label[n - 1 - global_idx]

        fwd = "".join(local_label_fwd)
        rev = "".join(local_label_rev)
        if rev.count("I") < fwd.count("I"):
            return rev
        return fwd

    @staticmethod
    def _compact_pauli_label(label: str) -> str:
        """Compact a Pauli label onto contiguous support by removing identity gaps."""
        # Stripping 'I's breaks positional mapping and causes REM matrix shape mismatch.
        # We must keep the 'I's to maintain alignment with active qubit measurements.
        return label

    @staticmethod
    def _to_mitiq_observable(observable: Any, circuit: Any = None) -> Any:
        """Convert Qiskit observables to Mitiq Observable when possible."""

        if isinstance(observable, MitiqObservable):
            return observable
        if isinstance(observable, MitiqPauliString):
            return MitiqObservable(observable)
        if isinstance(observable, (list, tuple, np.ndarray, ObservablesArray)):
            combined_paulis = []
            obs_items = np.asarray(observable, dtype=object).reshape(-1)
            for item in obs_items:
                converted = MNeat._to_mitiq_observable(item, circuit)
                if isinstance(converted, MitiqObservable):
                    combined_paulis.extend(converted.paulis)
                    continue
                raise ValueError(
                    f"Unsupported observable element type for array conversion: {type(item).__name__}."
                )
            if not combined_paulis:
                raise ValueError("Observable array has no convertible Pauli terms.")
            return MitiqObservable(*combined_paulis)
        if isinstance(observable, dict):
            pauli_terms = []
            for label, coeff in observable.items():
                projected_label = MNeat._project_pauli_label(str(label), circuit)
                compact_label = MNeat._compact_pauli_label(projected_label)
                coeff_complex = complex(coeff)
                if abs(coeff_complex) == 0:
                    continue
                pauli_terms.append(MitiqPauliString(spec=compact_label, coeff=coeff_complex))
            if not pauli_terms:
                raise ValueError("Observable mapping has no non-zero Pauli terms.")
            return MitiqObservable(*pauli_terms)
        if isinstance(observable, str):
            projected_label = MNeat._project_pauli_label(observable, circuit)
            compact_label = MNeat._compact_pauli_label(projected_label)
            return MitiqObservable(MitiqPauliString(spec=compact_label, coeff=1.0))
        if isinstance(observable, SparsePauliOp):
            pauli_terms = []
            for label, coeff in observable.to_list():
                projected_label = MNeat._project_pauli_label(label, circuit)
                compact_label = MNeat._compact_pauli_label(projected_label)
                coeff_complex = complex(coeff)
                if abs(coeff_complex) == 0:
                    continue
                pauli_terms.append(MitiqPauliString(spec=compact_label, coeff=coeff_complex))
            if not pauli_terms:
                raise ValueError("Observable has no non-zero Pauli terms.")
            return MitiqObservable(*pauli_terms)
        return observable

    def _make_scalar_estimator_executor(
        self,
        pub: EstimatorPub,
        observable: Any,
        with_noise: bool,
        seed_simulator: int | None,
        precision: float,
    ) -> Callable[[Any], float]:
        """Build a scalar expectation executor for one observable."""
        backend_options = {
            "method": "stabilizer",
            "noise_model": self.noise_model if with_noise else None,
            "seed_simulator": seed_simulator,
        }
        estimator = AerEstimator(
            options={"backend_options": backend_options, "default_precision": precision}
        )

        def executor(circuit) -> float:
            observables_for_pub: Any = np.asarray([observable], dtype=object)
            estimator_pub = EstimatorPub(
                circuit,
                observables_for_pub,
                pub.parameter_values,
                pub.precision,
                False,
            )
            data = estimator.run([estimator_pub]).result()[0].data
            evs = getattr(data, "evs", None)
            if evs is None:
                raise ValueError("Estimator result has no 'evs' field.")
            mean_ev = np.mean(evs)
            return self._extract_scalar(mean_ev)

        # With postponed annotations enabled, set runtime type explicitly for Mitiq.
        executor.__annotations__["return"] = float

        return executor

    def _make_sampler_measurement_executor(
        self,
        with_noise: bool,
        seed_simulator: int | None,
        shots: int,
    ) -> Callable[[QuantumCircuit], Any]:
        """Build a sampler executor returning MeasurementResult for sampler mode."""
        backend_options = {
            "method": "stabilizer",
            "noise_model": self.noise_model if with_noise else None,
            "seed_simulator": seed_simulator,
        }
        sampler = AerSampler(options={"backend_options": backend_options})

        from mitiq.typing import MeasurementResult as MitiqMeasurementResult

        def executor(circuit: QuantumCircuit) -> Any:

            if getattr(circuit, "num_clbits", 0) == 0:
                circuit = circuit.copy()
                active_qubits = self._active_qubit_indices(circuit)
                if active_qubits:
                    from qiskit import ClassicalRegister
                    c_reg = ClassicalRegister(len(active_qubits), name="meas")
                    circuit.add_register(c_reg)
                    for c_idx, q_idx in enumerate(active_qubits):
                        circuit.measure(circuit.qubits[q_idx], c_reg[c_idx])
            job = sampler.run([(circuit,)], shots=shots)
            data = job.result()[0].data
            counts = self._extract_counts_from_sampler_data(data)
            return MitiqMeasurementResult.from_counts(counts)

        # Mitiq infers executor output type from annotations when observables are provided.
        executor.__annotations__["return"] = MitiqMeasurementResult

        return executor
    
    def _get_mitiq_rem_confusion_matrix(self, circuit: QuantumCircuit) -> np.ndarray:
        """Generate a REM confusion matrix for the given circuit and noise model."""

        local_rem_matrices = []

        active_qubits = self._active_qubit_indices(circuit)
        
        for q in active_qubits:
            ro_error = self.noise_model._local_readout_errors.get((q,))
            if ro_error:
                a_matrix = np.array(ro_error.probabilities).T
                local_rem_matrices.append(a_matrix)
            else:
                local_rem_matrices.append(np.eye(2))
        
        if not local_rem_matrices:
            return np.eye(2 ** len(active_qubits))
        rem_inverse_confusion_matrix = generate_tensored_inverse_confusion_matrix(len(active_qubits), local_rem_matrices)

        return rem_inverse_confusion_matrix

    def _mitigate_value(
        self,
        technique: MitigationTechnique,
        circuit: QuantumCircuit,
        observable,
        noisy_executor,
        ideal_executor,
        execution_mode: ExecutionMode,
        technique_kwargs: dict[str, Any],
    ) -> Any:
        """Apply one mitigation technique to one circuit/executor pair."""
        kwargs = dict(technique_kwargs)
        mitiq_observable = (
            self._to_mitiq_observable(observable, circuit)
            if execution_mode == "sampler"
            else None
        )

        import inspect
        def filter_args(func, kw):
            spec = inspect.getfullargspec(func)
            if spec.varkw is not None:
                return dict(kw)
            return {k: v for k, v in kw.items() if k in spec.args + spec.kwonlyargs}

        if technique == "zne":
            f_kwargs = filter_args(zne.execute_with_zne, kwargs)
            if mitiq_observable is None:
                value = zne.execute_with_zne(
                    circuit,
                    noisy_executor,
                    **f_kwargs,
                )
            else:
                value = zne.execute_with_zne(
                    circuit,
                    noisy_executor,
                    observable=mitiq_observable,
                    **f_kwargs,
                )
            return value

        if technique == "pec":
            representations = kwargs.get("representations")
            if representations is None:
                noise_level = kwargs.get("noise_level", 0.01)
                representations = represent_ops_local(circuit, noise_level=noise_level)

            num_samples = kwargs.get("num_samples", 100)
            f_kwargs = filter_args(execute_with_pec, kwargs)
            for k in ["circuit", "executor", "observable", "representations", "num_samples", "noise_level"]:
                f_kwargs.pop(k, None)
            value = execute_with_pec(
                circuit,
                noisy_executor,
                observable=mitiq_observable,
                representations=representations,
                num_samples=num_samples,
                **f_kwargs,
            )
            return value

        if technique == "rem":
            inverse_confusion_matrix = kwargs.get("inverse_confusion_matrix")
            if inverse_confusion_matrix is None:
                inverse_confusion_matrix = self._get_mitiq_rem_confusion_matrix(circuit)

            f_kwargs = filter_args(execute_with_rem, kwargs)
            for k in ["circuit", "executor", "observable", "inverse_confusion_matrix"]:
                f_kwargs.pop(k, None)
            
            if mitiq_observable is None:
                raise ValueError("Warning: REM mitigation only works with an observable.")

            value = execute_with_rem(
                circuit,
                noisy_executor,
                mitiq_observable,
                inverse_confusion_matrix=inverse_confusion_matrix,
                **f_kwargs,
            )
            return value

        if technique == "cdr":
            simulator = kwargs.get("simulator", ideal_executor)
            if simulator is None:
                raise ValueError("CDR mitigation requires an ideal simulator callable.")
            num_training_circuits = kwargs.get("num_training_circuits", 24)
            fraction_non_clifford = kwargs.get("fraction_non_clifford", 0.1)
            
            f_kwargs = filter_args(execute_with_cdr, kwargs)
            for k in ["circuit", "executor", "observable", "simulator", "num_training_circuits", "fraction_non_clifford"]:
                f_kwargs.pop(k, None)

            value = execute_with_cdr(
                circuit,
                noisy_executor,
                observable=mitiq_observable,
                simulator=simulator,
                num_training_circuits=num_training_circuits,
                fraction_non_clifford=fraction_non_clifford,
                **f_kwargs,
            )
            return value

        if technique == "ddd":
            f_kwargs = filter_args(execute_with_ddd, kwargs)
            if "rule" not in f_kwargs:
                from mitiq.ddd.rules import xx
                f_kwargs["rule"] = xx # Default to XX rule if not specified
            value = execute_with_ddd(
                circuit,
                noisy_executor,
                observable=mitiq_observable,
                **f_kwargs,
            )
            return value

        if technique == "pea":
            if execution_mode == "sampler":
                raise ValueError(
                    "PEA requires a scalar executor. Use execution_mode='estimator'."
                )
            f_kwargs = filter_args(execute_with_pea, kwargs)
            for k in ["circuit", "executor", "observable"]:
                f_kwargs.pop(k, None)
            value = execute_with_pea(
                cast(Any, circuit),
                noisy_executor,
                **f_kwargs,
            )
            return value

        raise ValueError(
            "Unknown mitigation technique. Supported values are: 'zne', 'pec', 'rem', 'cdr', 'ddd', 'pea'."
        )

    def mitigated_sim(
        self,
        pubs: Sequence[EstimatorPubLike],
        technique: MitigationTechnique | Sequence[MitigationTechnique],
        cliffordize: bool = False,
        seed_simulator: int | None = None,
        precision: float = 0,
        execution_mode: ExecutionMode = "estimator",
        kwargs: dict[str, Any] | None = None,
    ) -> NeatResult:
        """Run a Mitiq-mitigated noisy simulation and return a NeatResult.

        Args:
            pubs: Estimator PUB inputs.
            technique: Mitigation method, one of ``"zne"``, ``"pec"``, ``"rem"``, ``"cdr"``, ``"ddd"``, ``"pea"``.
            cliffordize: If ``True``, convert PUB circuits with ``to_clifford`` before simulation.
            seed_simulator: Seed for Aer primitives.
            precision: Estimator precision used in ``execution_mode="fast"``.
            execution_mode:
                - ``"sampler"`` (default): sampler-based MeasurementResult executors.
                  Observables are forwarded to Mitiq.
                - ``"fast"``: estimator-based scalar executors.
            kwargs: Extra technique and executor kwargs passed to the selected Mitiq routine.

        Notes:
            Common kwargs by technique:
            - all native-mode techniques: ``sampler_shots`` (default: ``8192``)
            - zne: ``scale_noise``, ``factory``, ``num_to_average``
            - pec: ``representations`` or ``noise_level``, plus ``num_samples``
            - rem: ``inverse_confusion_matrix``
            - cdr: ``simulator``, ``num_training_circuits``, ``fraction_non_clifford``
            - ddd: ``rule``, ``rule_args``, ``num_trials``
            - pea: ``scale_factors``, ``noise_model``, ``epsilon``, ``extrapolation_method``, ``random_state``
        """
        self._ensure_mitiq()

        if cliffordize:
            coerced_pubs = self.to_clifford(pubs)
        else:
            coerced_pubs = [EstimatorPub.coerce(p) for p in pubs]

        cfg = dict(kwargs or {})
        sampler_shots = int(cfg.pop("sampler_shots", 8192))
        pub_results: list[NeatPubResult] = []

        for pub in coerced_pubs:
            obs_array = self._as_observable_array(pub.observables)

            if execution_mode == "sampler":
                noisy_executor = self._make_sampler_measurement_executor(
                    with_noise=True,
                    seed_simulator=seed_simulator,
                    shots=sampler_shots,
                )

            else:
                noisy_executor = self._make_scalar_estimator_executor(
                    pub=pub,
                    observable=obs_array,
                    with_noise=True,
                    seed_simulator=seed_simulator,
                    precision=precision,
                )

            ideal_executor = self._make_scalar_estimator_executor(
                    pub=pub,
                    observable=obs_array,
                    with_noise=False,
                    seed_simulator=seed_simulator,
                    precision=precision,
                )
            if isinstance(technique, str):
                techniques = [technique]
            else:
                techniques = list(technique)

            current_exec = noisy_executor
            current_mode = execution_mode

            for tech in techniques[:-1]:
                def make_wrapper(t: str, b_exec: Callable, c_mode: str, b_obs: Any):
                    def wrapped_executor(circ: QuantumCircuit) -> float:
                        temp_vals = self._mitigate_value(
                            technique=t, # type: ignore
                            circuit=circ,
                            observable=b_obs,
                            noisy_executor=b_exec,
                            ideal_executor=ideal_executor,
                            execution_mode=c_mode, # type: ignore
                            technique_kwargs=cfg,
                        )
                        temp_vals = np.asarray([np.real(temp_vals)], dtype=float)
                        if c_mode == "sampler" and b_obs is not None and len(b_obs) > 0:
                            temp_vals = temp_vals / len(b_obs)
                        return float(temp_vals)

                    wrapped_executor.__annotations__ = {'return': float}
                    return wrapped_executor

                current_exec = make_wrapper(tech, current_exec, current_mode, obs_array)
                current_mode = "estimator"
                obs_array = None

            val = self._mitigate_value(
                technique=techniques[-1], # type: ignore
                circuit=pub.circuit,
                observable=obs_array, # Will be None if nested!
                noisy_executor=current_exec,
                ideal_executor=ideal_executor,
                execution_mode=current_mode, # type: ignore
                technique_kwargs=cfg,
            )
            vals = np.asarray([np.real(val)], dtype=float)
            if execution_mode == "sampler" and obs_array is not None and len(obs_array) > 0:
                vals = vals / len(obs_array)
            pub_results.append(NeatPubResult(vals))

        return NeatResult(pub_results)
