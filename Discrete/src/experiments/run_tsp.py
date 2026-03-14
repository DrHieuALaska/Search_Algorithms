# DISCREATE PARAM TEST
# # src/experiments/run_tsp.py
from __future__ import annotations

import os
import sys
import argparse
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.logging import RunLogger
from core.config import load_yaml
from problems.tsp import TSPProblem
from solvers.sa_tsp import SimulatedAnnealingTSP
from solvers.tlbo_tsp import TLBO_TSP
from solvers.ga_tsp import GeneticAlgorithmTSP
from solvers.hc_tsp import HC_TSP
from solvers.aco_tsp import ACO_TSP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TSP experiments from YAML config")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., config_tsp_param.yaml)")
    return p.parse_args()


# def main():
#     args = parse_args()
#     cfg = load_yaml(args.config)

#     exp = cfg.get("experiment", {})
#     algos = cfg.get("algos", ["SA"])

#     # experiment settings
#     exp_id = exp.get("exp_id", "exp_001")
#     n_list = exp.get("n_list", [5, 15, 30])
#     instances = int(exp.get("instances", 10))
#     runs_per_instance = int(exp.get("runs_per_instance", 10))
#     scale = float(exp.get("scale", 100.0))

#     run_csv = exp.get("run_csv", "runs.csv")
#     trace_csv = exp.get("trace_csv", "trace.csv")
#     code_version = exp.get("code_version", "")

#     logger = RunLogger(run_csv=run_csv, trace_csv=trace_csv)

#     solvers = {}

#     if "SA" in algos:
#         p = cfg.get("SA", {})
#         solvers["SA"] = SimulatedAnnealingTSP(
#             T0=float(p.get("T0", 100.0)),
#             Tmin=float(p.get("Tmin", 1e-3)),
#             alpha=float(p.get("alpha", 0.99)),
#             steps_per_temp=int(p.get("steps_per_temp", 2000)),
#             max_iter=int(p.get("max_iter", 50000)),
#             trace_every=int(p.get("trace_every", 200)),
#         )

#     if "TLBO" in algos:
#         p = cfg.get("TLBO", {})
#         solvers["TLBO"] = TLBO_TSP(
#             pop_size=int(p.get("pop_size", 50)),
#             iters=int(p.get("iters", 500)),
#             trace_every=int(p.get("trace_every", 10)),
#         )

#     if "GA" in algos:
#         p = cfg.get("GA", {})
#         solvers["GA"] = GeneticAlgorithmTSP(
#             population_size=int(p.get("population_size", 100)),
#             generations=int(p.get("generations", 400)),
#             mutation_rate=float(p.get("mutation_rate", 0.2)),
#             tournament_k=int(p.get("tournament_k", 10)),
#             trace_every=int(p.get("trace_every", 10)),
#         )

#     if "HC" in algos:
#         p = cfg.get("HC", {})
#         solvers["HC"] = HC_TSP(
#             max_iter=int(p.get("max_iter", 50000)),
#             mode=str(p.get("mode", "first")),
#             trace_every=int(p.get("trace_every", 200)),
#         )

#     if "ACO" in algos:
#         p = cfg.get("ACO", {})
#         solvers["ACO"] = ACO_TSP(
#             n_ants=int(p.get("n_ants", 30)),
#             iters=int(p.get("iters", 1667)),
#             alpha=float(p.get("alpha", 1.0)),
#             beta=float(p.get("beta", 3.0)),
#             rho=float(p.get("rho", 0.1)),
#             Q=float(p.get("Q", 100.0)),
#             trace_every=int(p.get("trace_every", 10)),
#             deposit_best_only=bool(p.get("deposit_best_only", False)),
#         )

#     for n in n_list:
#         for inst in range(instances):
#             problem = TSPProblem(n_cities=int(n), instance_seed=inst, scale=scale)

#             for run in range(runs_per_instance):
#                 for algo_name, solver in solvers.items():
#                     run_id = f"{algo_name}_n{n}_inst{inst}_run{run}"
#                     solver.solve(
#                         problem,
#                         init_solution=None,
#                         logger=logger,
#                         experiment_id=exp_id,
#                         run_id=run_id,
#                         instance_seed=inst,
#                         seed_algo=run,
#                         coord_scale=scale,
#                         code_version=code_version,
#                     )

#     print("Done.")
#     print(f"Run log:   {run_csv}")
#     print(f"Trace log: {trace_csv}")



# MULTIPLE PARAM TEST
import itertools
import math

def _grid(sweep_dict: dict) -> list[dict]:
    """Cartesian product of sweep parameters."""
    if not sweep_dict:
        return [{}]
    keys = list(sweep_dict.keys())
    values = [sweep_dict[k] for k in keys]
    combos = []
    for prod in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos

def _tag(params: dict) -> str:
    """Short tag for run_id (paper-friendly)."""
    if not params:
        return "base"
    parts = []
    for k, v in params.items():
        kk = k.replace("_", "")
        parts.append(f"{kk}{v}")
    return "-".join(parts)

def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    exp = cfg.get("experiment", {})
    algos = cfg.get("algos", ["SA"])
    sweeps = cfg.get("sweeps", {})

    exp_id = exp.get("exp_id", "exp_001")
    n_list = exp.get("n_list", [5, 15, 30, 50])
    instances = int(exp.get("instances", 10))
    runs_per_instance = int(exp.get("runs_per_instance", 10))
    scale = float(exp.get("scale", 100.0))
    budget = int(exp.get("budget_evals", 50000))  # << normalize effort

    run_csv = exp.get("run_csv", "runs.csv")
    trace_csv = exp.get("trace_csv", "trace.csv")
    code_version = exp.get("code_version", "")

    logger = RunLogger(run_csv=run_csv, trace_csv=trace_csv)

    # Build all solver variants per algorithm
    solver_variants = []  # list of (algo_name, tag, solver_obj, extra_meta)
    for algo in algos:
        base_params = cfg.get(algo, {}) if isinstance(cfg.get(algo, {}), dict) else {}
        sweep_params = sweeps.get(algo, {}) if isinstance(sweeps.get(algo, {}), dict) else {}

        for sparams in _grid(sweep_params):
            params = dict(base_params)
            params.update(sparams)
            tag = _tag(sparams)

            extra_meta = {"param_tag": tag}  # optional: stored in RunResult.meta

            # --- SA: enforce budget via max_iter ---
            if algo == "SA":
                T0 = float(params.get("T0", 100.0))
                Tmin = float(params.get("Tmin", 1e-3))
                alpha = float(params.get("alpha", 0.99))
                steps_per_temp = int(params.get("steps_per_temp", 2000))
                trace_every = int(params.get("trace_every", 200))
                max_iter = budget - 1  # ~1 eval per iter + 1 init eval

                solver = SimulatedAnnealingTSP(
                    T0=T0, Tmin=Tmin, alpha=alpha,
                    steps_per_temp=steps_per_temp,
                    max_iter=max_iter,
                    trace_every=trace_every
                )
                solver_variants.append((algo, tag, solver, extra_meta))

            # --- GA: match budget by choosing generations ---
            elif algo == "GA":
                pop = int(params.get("population_size", 100))
                mut = float(params.get("mutation_rate", 0.2))
                k = int(params.get("tournament_k", 10))
                trace_every = int(params.get("trace_every", 10))

                gens = max(1, (budget - pop) // pop)  # evals ≈ pop + gens*pop

                solver = GeneticAlgorithmTSP(
                    population_size=pop,
                    generations=gens,
                    mutation_rate=mut,
                    tournament_k=k,
                    trace_every=trace_every
                )
                solver_variants.append((algo, tag, solver, extra_meta))

            # --- ACO: exact budget match by choosing iters ---
            elif algo == "ACO":
                ants = int(params.get("n_ants", 30))
                aco_iters = max(1, math.ceil(budget / ants))  # evals = ants*iters

                solver = ACO_TSP(
                    n_ants=ants,
                    iters=aco_iters,
                    alpha=float(params.get("alpha", 1.0)),
                    beta=float(params.get("beta", 3.0)),
                    rho=float(params.get("rho", 0.1)),
                    Q=float(params.get("Q", 100.0)),
                    trace_every=int(params.get("trace_every", 10)),
                    deposit_best_only=bool(params.get("deposit_best_only", False)),
                )
                solver_variants.append((algo, tag, solver, extra_meta))

            # --- HC: implement restarts at runner level ---
            elif algo == "HC":
                mode = str(params.get("mode", "first"))
                restarts = int(params.get("restarts", 1))
                trace_every = int(params.get("trace_every", 200))

                # We will run HC multiple times and keep best; budget split across restarts
                # We'll store restarts in extra meta; actual HC solver constructed per restart
                extra_meta = {"param_tag": tag, "hc_mode": mode, "hc_restarts": restarts, "hc_trace_every": trace_every}
                solver_variants.append((algo, tag, None, extra_meta))

            # --- TLBO: keep your params, but recommend budget-capping in solver later ---
            elif algo == "TLBO":
                solver = TLBO_TSP(
                    pop_size=int(params.get("pop_size", 50)),
                    iters=int(params.get("iters", 500)),
                    trace_every=int(params.get("trace_every", 10)),
                    move_frac_max=float(params.get("move_frac_max", 1.0)),
                )
                solver_variants.append((algo, tag, solver, extra_meta))

    # --- Run experiments ---
    for n in n_list:
        for inst in range(instances):
            problem = TSPProblem(n_cities=int(n), instance_seed=inst, scale=scale)

            for run in range(runs_per_instance):
                for algo_name, tag, solver, extra in solver_variants:
                    run_id = f"{algo_name}[{tag}]_n{n}_inst{inst}_run{run}"

                    # HC special: restarts wrapper
                    if algo_name == "HC":
                        mode = extra["hc_mode"]
                        restarts = extra["hc_restarts"]
                        trace_every = extra["hc_trace_every"]
                        per_restart = max(1, (budget - 1) // restarts)

                        # Run restarts and keep best (simple; logs run-level via logger)
                        best_cost = float("inf")
                        best_sol = None
                        evals_total = 0
                        iters_total = 0
                        t0 = time.perf_counter()

                        for r in range(restarts):
                            hc = HC_TSP(max_iter=per_restart, mode=mode, trace_every=trace_every)
                            # shift seed for each restart
                            rr_seed = run * 1000 + r
                            res, _ = hc.solve(
                                problem,
                                init_solution=None,
                                logger=None,  # avoid per-restart log spam
                                experiment_id=exp_id,
                                run_id=run_id,
                                instance_seed=inst,
                                seed_algo=rr_seed,
                                coord_scale=scale,
                                code_version=code_version,
                            )
                            evals_total += res.evals_cost
                            iters_total += res.iters
                            if res.best_cost < best_cost:
                                best_cost = res.best_cost
                                best_sol = res.best_solution

                        time_sec = time.perf_counter() - t0

                        # One consolidated log row for HC restarts (fills shared schema)
                        if logger is not None:
                            timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                            logger.log_run({
                                "experiment_id": exp_id,
                                "run_id": run_id,
                                "timestamp_utc": timestamp_utc,
                                "algorithm": "HC",
                                "problem": "TSP",
                                "n_cities": int(n),
                                "instance_seed": inst,
                                "coord_scale": scale,
                                "distance_type": "euclidean",
                                "init_temp_T0": "",
                                "min_temp_Tmin": "",
                                "alpha": "",
                                "steps_per_temp": "",
                                "max_iter": budget - 1,
                                "neighbor_operator": f"2-opt-{mode}-restarts{restarts}",
                                "iters": iters_total,
                                "evals_cost": evals_total,
                                "time_sec": time_sec,
                                "init_cost": "",
                                "final_cost": best_cost,
                                "best_cost": best_cost,
                                "iter_best_found": "",
                                "evals_best_found": "",
                                "time_best_found_sec": "",
                                "seed_algo": run,
                                "code_version": code_version,
                            })
                        continue

                    # Normal solvers
                    solver.solve(
                        problem,
                        init_solution=None,
                        logger=logger,
                        experiment_id=exp_id,
                        run_id=run_id,
                        instance_seed=inst,
                        seed_algo=run,
                        coord_scale=scale,
                        code_version=code_version,
                        **(extra or {})
                    )

    print("Done.")
    print(f"Run log:   {run_csv}")
    print(f"Trace log: {trace_csv}")

if __name__ == "__main__":
    main()