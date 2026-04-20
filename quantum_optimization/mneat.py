"""MNEAT: Mitigated Noise Estimator Analyzer Tool.

This module extends Qiskit Runtime's NEAT class with Mitiq-based mitigation
simulation methods that preserve the NEAT return type (NeatResult).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from qiskit.primitives.containers import EstimatorPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit_ibm_runtime.debug_tools import Neat
from qiskit_ibm_runtime.debug_tools.neat_results import NeatPubResult, NeatResult

try:
    from qiskit_aer.primitives.estimator_v2 import EstimatorV2 as AerEstimator
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

MitigationTechnique = Literal["zne", "pec", "rem", "cdr"]


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

    def _make_scalar_estimator_executor(
        self,
        pub: EstimatorPub,
        observable: Any,
        with_noise: bool,
        seed_simulator: int | None,
        precision: float,
    ):
        """Build a scalar expectation executor for one observable."""
        backend_options = {
            "method": "stabilizer",
            "noise_model": self.noise_model if with_noise else None,
            "seed_simulator": seed_simulator,
        }
        estimator = AerEstimator(
            options={"backend_options": backend_options, "default_precision": precision}
        )

        def executor(circuit):
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

    def _mitigate_value(
        self,
        technique: MitigationTechnique,
        circuit,
        observable,
        noisy_executor,
        ideal_executor,
        technique_kwargs: dict[str, Any],
    ) -> float:
        """Apply one mitigation technique to one circuit/executor pair."""
        kwargs = dict(technique_kwargs)

        if technique == "zne":
            value = zne.execute_with_zne(
                circuit,
                noisy_executor,
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
        technique_kwargs: dict[str, Any] | None = None,
    ) -> NeatResult:
        """Run a Mitiq-mitigated noisy simulation and return a NeatResult."""
        self._ensure_mitiq()

        if cliffordize:
            coerced_pubs = self.to_clifford(pubs)
        else:
            coerced_pubs = [EstimatorPub.coerce(p) for p in pubs]

        cfg = technique_kwargs or {}
        pub_results: list[NeatPubResult] = []

        for pub in coerced_pubs:
            obs_array = self._as_observable_array(pub.observables)
            flat_obs = obs_array.reshape(-1)
            mitigated_vals: list[float] = []

            for obs in flat_obs:
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
        **zne_kwargs: Any,
    ) -> NeatResult:
        """Convenience wrapper for mitigated_sim(..., technique='zne')."""
        return self.mitigated_sim(
            pubs=pubs,
            technique="zne",
            cliffordize=cliffordize,
            seed_simulator=seed_simulator,
            precision=precision,
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
