"""MNEAT: Mitigated Noise Estimator Analyzer Tool.

This module extends Qiskit Runtime's NEAT class with Mitiq-based mitigation
simulation methods that preserve the NEAT return type (NeatResult).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives.containers import EstimatorPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub
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
    from mitiq.cdr import execute_with_cdr
    from mitiq.pec import (
        execute_with_pec,
        represent_operations_in_circuit_with_local_depolarizing_noise as represent_ops_local,
    )
    from mitiq.rem import execute_with_rem
    from mitiq.zne.inference import RichardsonFactory

    HAS_MITIQ = True
except ImportError:
    HAS_MITIQ = False

if TYPE_CHECKING:
    from mitiq import MeasurementResult as MitiqMeasurementResult
else:
    MitiqMeasurementResult = Any

MitigationTechnique = Literal["zne", "pec", "rem", "cdr"]
ExecutionMode = Literal["fast", "mitiq-native"]


class MNeat(Neat):
    """NEAT extension with Mitiq-backed mitigated noisy simulations.

    Core API:
      - mitigated_sim(..., technique=..., technique_kwargs=...)

    Convenience wrappers:
      - zne_mit_noisy_sim(...)
      - pec_mit_noisy_sim(...)
      - rem_mit_noisy_sim(...)
      - cdr_mit_noisy_sim(...)
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
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size != 1:
            raise ValueError(
                "Mitigation executor returned a non-scalar value. Use scalar PUB expectations."
            )
        return float(arr[0])

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
    def _to_mitiq_observable(observable: Any) -> Any:
        """Convert Qiskit observables to Mitiq Observable when possible."""
        try:
            from mitiq import Observable as MitiqObservable
            from mitiq import PauliString as MitiqPauliString
        except ImportError:
            return observable

        if isinstance(observable, MitiqObservable):
            return observable
        if isinstance(observable, SparsePauliOp):
            pauli_terms = []
            for label, coeff in observable.to_list():
                coeff_complex = complex(coeff)
                if abs(coeff_complex) == 0:
                    continue
                pauli_terms.append(MitiqPauliString(spec=label, coeff=coeff_complex))
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
            trial_pub = EstimatorPub(
                circuit,
                observables_for_pub,
                pub.parameter_values,
                pub.precision,
                False,
            )
            data = estimator.run([trial_pub]).result()[0].data
            evs = getattr(data, "evs", None)
            if evs is None:
                raise ValueError("Estimator result has no 'evs' field.")
            return self._extract_scalar(evs)

        return executor

    def _make_sampler_measurement_executor(
        self,
        with_noise: bool,
        seed_simulator: int | None,
        shots: int,
    ) -> Callable[[QuantumCircuit], MitiqMeasurementResult]:
        """Build a sampler executor returning MeasurementResult for Mitiq-native mode."""
        backend_options = {
            "method": "stabilizer",
            "noise_model": self.noise_model if with_noise else None,
            "seed_simulator": seed_simulator,
        }
        sampler = AerSampler(options={"backend_options": backend_options})

        try:
            pass_manager = generate_preset_pass_manager(
                optimization_level=0,
                backend=self.backend,
            )
        except Exception:
            pass_manager = None

        def executor(circuit: QuantumCircuit) -> MitiqMeasurementResult:
            from mitiq import MeasurementResult as RuntimeMeasurementResult

            exec_circuit = pass_manager.run(circuit) if pass_manager is not None else circuit
            if getattr(exec_circuit, "num_clbits", 0) == 0:
                exec_circuit = exec_circuit.copy()
                exec_circuit.measure_all()
            job = sampler.run([(exec_circuit,)], shots=shots)
            data = job.result()[0].data
            counts = self._extract_counts_from_sampler_data(data)
            return RuntimeMeasurementResult.from_counts(counts)

        return executor

    def _mitigate_value(
        self,
        technique: MitigationTechnique,
        circuit,
        observable,
        noisy_executor,
        ideal_executor,
        execution_mode: ExecutionMode,
        technique_kwargs: dict[str, Any],
    ) -> float:
        """Apply one mitigation technique to one circuit/executor pair."""
        kwargs = dict(technique_kwargs)

        if technique == "zne":
            zne_observable = None
            if execution_mode == "mitiq-native":
                zne_observable = self._to_mitiq_observable(observable)
            value = zne.execute_with_zne(
                circuit,
                noisy_executor,
                observable=zne_observable,
                **kwargs,
            )
            return self._extract_scalar(value)

        if technique == "pec":
            representations = kwargs.pop("representations", None)
            if representations is None:
                noise_level = kwargs.pop("noise_level", 0.01)
                representations = represent_ops_local(circuit, noise_level=noise_level)

            num_samples = kwargs.pop("num_samples", 100)
            value = execute_with_pec(
                circuit,
                noisy_executor,
                representations=representations,
                num_samples=num_samples,
                **kwargs,
            )
            return self._extract_scalar(value)

        if technique == "rem":
            inverse_confusion_matrix = kwargs.pop("inverse_confusion_matrix", None)
            value = execute_with_rem(
                circuit,
                noisy_executor,
                observable,
                inverse_confusion_matrix=inverse_confusion_matrix,
                **kwargs,
            )
            return self._extract_scalar(value)

        if technique == "cdr":
            simulator = kwargs.pop("simulator", ideal_executor)
            if simulator is None:
                raise ValueError("CDR mitigation requires an ideal simulator callable.")
            num_training_circuits = kwargs.pop("num_training_circuits", 24)
            fraction_non_clifford = kwargs.pop("fraction_non_clifford", 0.1)
            value = execute_with_cdr(
                circuit,
                noisy_executor,
                simulator=simulator,
                num_training_circuits=num_training_circuits,
                fraction_non_clifford=fraction_non_clifford,
                **kwargs,
            )
            return self._extract_scalar(value)

        raise ValueError(
            "Unknown mitigation technique. Supported values are: 'zne', 'pec', 'rem', 'cdr'."
        )

    def mitigated_sim(
        self,
        pubs: Sequence[EstimatorPubLike],
        technique: MitigationTechnique,
        cliffordize: bool = False,
        seed_simulator: int | None = None,
        precision: float = 0,
        execution_mode: ExecutionMode = "fast",
        technique_kwargs: dict[str, Any] | None = None,
    ) -> NeatResult:
        """Run a Mitiq-mitigated noisy simulation and return a NeatResult."""
        self._ensure_mitiq()
        if execution_mode == "mitiq-native" and technique != "zne":
            raise ValueError(
                "Execution mode 'mitiq-native' is currently supported only for technique='zne'."
            )

        if cliffordize:
            coerced_pubs = self.to_clifford(pubs)
        else:
            coerced_pubs = [EstimatorPub.coerce(p) for p in pubs]

        cfg = dict(technique_kwargs or {})
        sampler_shots = int(cfg.pop("sampler_shots", 8192))
        pub_results: list[NeatPubResult] = []

        for pub in coerced_pubs:
            obs_array = self._as_observable_array(pub.observables)
            flat_obs = obs_array.reshape(-1)
            mitigated_vals: list[float] = []
            native_noisy_executor = None

            if execution_mode == "mitiq-native":
                native_noisy_executor = self._make_sampler_measurement_executor(
                    with_noise=True,
                    seed_simulator=seed_simulator,
                    shots=sampler_shots,
                )

            for obs in flat_obs:
                if execution_mode == "mitiq-native":
                    noisy_executor = native_noisy_executor
                    ideal_executor = None
                else:
                    noisy_executor = self._make_scalar_estimator_executor(
                        pub=pub,
                        observable=obs,
                        with_noise=True,
                        seed_simulator=seed_simulator,
                        precision=precision,
                    )
                    ideal_executor = self._make_scalar_estimator_executor(
                        pub=pub,
                        observable=obs,
                        with_noise=False,
                        seed_simulator=seed_simulator,
                        precision=precision,
                    )
                val = self._mitigate_value(
                    technique=technique,
                    circuit=pub.circuit,
                    observable=obs,
                    noisy_executor=noisy_executor,
                    ideal_executor=ideal_executor,
                    execution_mode=execution_mode,
                    technique_kwargs=cfg,
                )
                mitigated_vals.append(val)

            vals = np.asarray(mitigated_vals, dtype=float).reshape(obs_array.shape)
            pub_results.append(NeatPubResult(vals))

        return NeatResult(pub_results)

    def zne_mit_noisy_sim(
        self,
        pubs: Sequence[EstimatorPubLike],
        cliffordize: bool = False,
        seed_simulator: int | None = None,
        precision: float = 0,
        execution_mode: ExecutionMode = "fast",
        **zne_kwargs: Any,
    ) -> NeatResult:
        """Convenience wrapper for mitigated_sim(..., technique='zne')."""
        return self.mitigated_sim(
            pubs=pubs,
            technique="zne",
            cliffordize=cliffordize,
            seed_simulator=seed_simulator,
            precision=precision,
            execution_mode=execution_mode,
            technique_kwargs=zne_kwargs,
        )

    def pec_mit_noisy_sim(
        self,
        pubs: Sequence[EstimatorPubLike],
        cliffordize: bool = False,
        seed_simulator: int | None = None,
        precision: float = 0,
        **pec_kwargs: Any,
    ) -> NeatResult:
        """Convenience wrapper for mitigated_sim(..., technique='pec')."""
        return self.mitigated_sim(
            pubs=pubs,
            technique="pec",
            cliffordize=cliffordize,
            seed_simulator=seed_simulator,
            precision=precision,
            technique_kwargs=pec_kwargs,
        )

    def rem_mit_noisy_sim(
        self,
        pubs: Sequence[EstimatorPubLike],
        cliffordize: bool = False,
        seed_simulator: int | None = None,
        precision: float = 0,
        **rem_kwargs: Any,
    ) -> NeatResult:
        """Convenience wrapper for mitigated_sim(..., technique='rem')."""
        return self.mitigated_sim(
            pubs=pubs,
            technique="rem",
            cliffordize=cliffordize,
            seed_simulator=seed_simulator,
            precision=precision,
            technique_kwargs=rem_kwargs,
        )

    def cdr_mit_noisy_sim(
        self,
        pubs: Sequence[EstimatorPubLike],
        cliffordize: bool = False,
        seed_simulator: int | None = None,
        precision: float = 0,
        **cdr_kwargs: Any,
    ) -> NeatResult:
        """Convenience wrapper for mitigated_sim(..., technique='cdr')."""
        return self.mitigated_sim(
            pubs=pubs,
            technique="cdr",
            cliffordize=cliffordize,
            seed_simulator=seed_simulator,
            precision=precision,
            technique_kwargs=cdr_kwargs,
        )
