
import logging
from typing import Type, Optional

from pyDOE import lhs 
from qiskit import QuantumCircuit
from qiskit.primitives import BindingsArrayLike, BaseEstimatorV2
from qiskit.providers import Backend
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import SPSA
from qiskit_ibm_runtime import Session
from scipy.optimize import minimize

from pipeline.utils import OptimizationCache

logger = logging.getLogger("pipeline_logger")


def cost_estimator(
        x,
        ansatz,
        hamiltonian,
        estimator,
        objective_vals,
        cache: Optional[OptimizationCache],
        run_id: int
):

    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, x)
    job = estimator.run([pub])
    results = job.result()[0]
    cost = float(results.data.evs)

    objective_vals.append(cost)
    if cache:
        cache.update_run(run_id, x.tolist(), cost)

    return cost


def parameter_optimization(
        n_layer: int,
        n_starting_point: int,
        bounds: tuple[float, float],
        optimization_params: dict,
        Estimator: type,
        estimator_shots: int,
        backend,
        circuit,
        hamiltonian,
        use_cache: bool,
        cache_filename: str,
        cache_save_every: int = 1
):

    cache = OptimizationCache(cache_filename, cache_save_every) if use_cache else None

    lower_bound, upper_bound = bounds
    lhs_samples = lhs(2 * n_layer, samples=n_starting_point, criterion='center')
    scaled_samples = lhs_samples * (upper_bound - lower_bound) + lower_bound

    best_cost = float('inf')
    best_result = None
    objective_fun_vals_all = []
    final_params_all = []

    for run in range(n_starting_point):

        if use_cache and cache.is_completed(run):
            logger.info(f"Run {run + 1} already completed. Skipping.")
            continue

        logger.info(f"Starting optimization run {run+1}/{n_starting_point}")
        run_data = cache.get_run(run) if use_cache else None

        if run_data is not None:
            init_params = run_data.get("best_params") or run_data.get("initial_params")
            objective_vals = run_data.get("objective_values", [])
        else:
            init_params = scaled_samples[run, :].tolist()
            objective_vals = []
            if use_cache:
                cache.start_run(run, init_params)

        with Session(backend=backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = estimator_shots

            if optimization_params['optimizer'] == "COBYLA":

                result = minimize(
                    cost_estimator,
                    init_params,
                    args=(circuit, hamiltonian, estimator, objective_vals, cache, run),
                    method='COBYLA',
                    tol=optimization_params['tolerance'],
                    options={'maxiter': 1000}
                )

            elif optimization_params['optimizer'] == "SPSA":

                optimizer = SPSA(
                    maxiter=optimization_params['maxiter'],
                    learning_rate=optimization_params['learning_rate'],
                    perturbation=optimization_params['perturbation']
                )

                # maxiter = int(optimization_params['maxiter'])
                # a0 = float(optimization_params['learning_rate'])
                # c0 = float(optimization_params['perturbation'])
                # alpha = float(optimization_params.get('learning_rate_decay', 0.602))
                # gamma = float(optimization_params.get('perturbation_decay', 0.101))
                # A = int(optimization_params.get('stability_constant', max(1, int(0.1 * maxiter))))

                # k = np.arange(maxiter, dtype=float)
                # lr_schedule = a0 / np.power(k + 1.0 + A, alpha)
                # pert_schedule = c0 / np.power(k + 1.0, gamma)

                # optimizer = SPSA(
                #     maxiter=maxiter,
                #     learning_rate=lr_schedule,
                #     perturbation=pert_schedule,
                #     trust_region=bool(optimization_params.get('trust_region', False)),
                #     blocking=bool(optimization_params.get('blocking', False)),
                #     last_avg=optimization_params.get('last_avg', None)
                # )

                def objective_wrapper(params):
                    return cost_estimator(params, circuit, hamiltonian, estimator, objective_vals, cache, run)

                result = optimizer.minimize(
                    fun=objective_wrapper,
                    x0=init_params
                )

            else:
                raise ValueError("Not a valid optimizer.")

            num_estimator_calls = result.nfev
            logger.info(f"Run {run + 1}: Estimator calls = {num_estimator_calls}")

            if use_cache:
                cache.complete_run(run)

            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result

            objective_fun_vals_all.append(objective_vals)
            final_params_all.append(result.x.tolist())


    if best_result is None and use_cache:

        completed_runs = cache.get_all_completed()

        if not completed_runs:
            raise RuntimeError("All runs cached but no valid results found.")
        best = min(completed_runs, key=lambda r: r["best_cost"])

        objective_fun_vals_all = [r.get("objective_values", []) for r in completed_runs]
        final_params_all = [r.get("best_params") for r in completed_runs]

        return best["best_params"], objective_fun_vals_all, final_params_all

    return best_result.x, objective_fun_vals_all, final_params_all



def sample_circuit(circuit: QuantumCircuit, backend: Backend, Sampler: Type, n_shot: int) -> dict:

    sampler = Sampler(mode=backend)

    pub = (circuit,)
    job = sampler.run([pub], shots=n_shot)

    counts_bin = job.result()[0].data.classic_register.get_counts()

    shots = sum(counts_bin.values())

    return {
        key[::-1]: val / shots
        for key, val in sorted(counts_bin.items(), key=lambda item: item[1] / shots, reverse=True)
    }