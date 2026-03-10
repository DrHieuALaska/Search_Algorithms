from __future__ import annotations

import os
import sys
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.logging import RunLogger
from core.config import load_yaml
from problems.knapsack import KnapsackProblem

from solvers.knapsack.sa_knapsack import SimulatedAnnealingKnapsack
from solvers.knapsack.ga_knapsack import GeneticAlgorithmKnapsack
from solvers.knapsack.hc_knapsack import HC_Knapsack
from solvers.knapsack.tlbo_knapsack import TLBO_Knapsack
from solvers.knapsack.abc_knapsack import ABC_Knapsack


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Knapsack experiments from YAML config")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., config_knapsack.yaml)")
    return p.parse_args()


# def main():
#     args = parse_args()
#     cfg = load_yaml(args.config)

#     exp = cfg.get("experiment", {})
#     algos = cfg.get("algos", ["SA_KP"])

#     exp_id = exp.get("exp_id", "exp_kp_001")
#     n_items_list = exp.get("n_items_list", [50, 100, 200])
#     instances = int(exp.get("instances", 10))
#     runs_per_instance = int(exp.get("runs_per_instance", 10))

#     # instance generation params
#     weight_low = int(exp.get("weight_low", 1))
#     weight_high = int(exp.get("weight_high", 50))
#     value_low = int(exp.get("value_low", 1))
#     value_high = int(exp.get("value_high", 100))
#     capacity_ratio = float(exp.get("capacity_ratio", 0.5))

#     penalty_lambda = float(exp.get("penalty_lambda", 1000.0))
#     feasible_only = bool(exp.get("feasible_only", True))

#     run_csv = exp.get("run_csv", "runs_knapsack.csv")
#     trace_csv = exp.get("trace_csv", "trace_knapsack.csv")
#     code_version = exp.get("code_version", "")

#     logger = RunLogger(run_csv=run_csv, trace_csv=trace_csv)

#     solvers = {}

#     if "SA_KP" in algos:
#         p = cfg.get("SA_KP", {})
#         solvers["SA_KP"] = SimulatedAnnealingKnapsack(
#             T0=float(p.get("T0", 10.0)),
#             Tmin=float(p.get("Tmin", 1e-3)),
#             alpha=float(p.get("alpha", 0.99)),
#             max_iter=int(p.get("max_iter", 50000)),
#             trace_every=int(p.get("trace_every", 200)),
#             feasible_only=feasible_only,
#         )

#     if "GA_KP" in algos:
#         p = cfg.get("GA_KP", {})
#         solvers["GA_KP"] = GeneticAlgorithmKnapsack(
#             population_size=int(p.get("population_size", 100)),
#             generations=int(p.get("generations", 400)),
#             crossover_rate=float(p.get("crossover_rate", 0.9)),
#             mutation_rate=float(p.get("mutation_rate", 0.02)),
#             tournament_k=int(p.get("tournament_k", 10)),
#             trace_every=int(p.get("trace_every", 10)),
#             penalty_lambda=penalty_lambda,
#             feasible_only=feasible_only,
#         )

#     if "HC_KP" in algos:
#         p = cfg.get("HC_KP", {})
#         solvers["HC_KP"] = HC_Knapsack(
#             max_iter=int(p.get("max_iter", 50000)),
#             mode=str(p.get("mode", "first")),
#             trace_every=int(p.get("trace_every", 200)),
#             flip_k=int(p.get("flip_k", 1)),
#             penalty_lambda=penalty_lambda,
#             feasible_only=feasible_only,
#         )

#     if "TLBO_KP" in algos:
#         p = cfg.get("TLBO_KP", {})
#         solvers["TLBO_KP"] = TLBO_Knapsack(
#             pop_size=int(p.get("pop_size", 50)),
#             iters=int(p.get("iters", 500)),
#             trace_every=int(p.get("trace_every", 10)),
#             penalty_lambda=penalty_lambda,
#             feasible_only=feasible_only,
#         )

#     if "ABC_KP" in algos:
#         p = cfg.get("ABC_KP", {})
#         solvers["ABC_KP"] = ABC_Knapsack(
#             sn=int(p.get("sn", 50)),
#             iters=int(p.get("iters", 500)),
#             limit=int(p.get("limit", 50)),
#             trace_every=int(p.get("trace_every", 10)),
#             flip_k=int(p.get("flip_k", 1)),
#             penalty_lambda=penalty_lambda,
#             feasible_only=feasible_only,
#         )

#     # Run experiments
#     for n_items in n_items_list:
#         for inst in range(instances):
#             # Create an instance (adjust if your API differs)
#             problem = KnapsackProblem.random_instance(
#                 int(n_items),
#                 instance_seed=inst,
#                 weight_range=(weight_low, weight_high),
#                 value_range=(value_low, value_high),
#                 capacity_ratio=capacity_ratio,
#             )

#             for run in range(runs_per_instance):
#                 for algo_name, solver in solvers.items():
#                     run_id = f"{algo_name}_n{n_items}_inst{inst}_run{run}"
#                     solver.solve(
#                         problem,
#                         init_solution=None,
#                         logger=logger,
#                         experiment_id=exp_id,
#                         run_id=run_id,
#                         instance_seed=inst,
#                         seed_algo=run,
#                         code_version=code_version,
#                     )

#     print("Done.")
#     print(f"Run log:   {run_csv}")
#     print(f"Trace log: {trace_csv}")


import itertools
import math

def _grid(sweep_dict: dict) -> list[dict]:
    if not sweep_dict:
        return [{}]
    keys = list(sweep_dict.keys())
    values = [sweep_dict[k] for k in keys]
    return [{k: v for k, v in zip(keys, prod)} for prod in itertools.product(*values)]

def _tag(params: dict) -> str:
    if not params:
        return "base"
    parts = []
    for k, v in params.items():
        kk = k.replace("_", "")
        v_str = str(v).replace(".", "p")
        parts.append(f"{kk}{v_str}")
    return "-".join(parts)

def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    exp = cfg.get("experiment", {})
    algos = cfg.get("algos", ["SA_KP"])
    sweeps = cfg.get("sweeps", {})

    exp_id = exp.get("exp_id", "exp_kp_001")
    n_items_list = exp.get("n_items_list", [100])
    instances = int(exp.get("instances", 10))
    runs_per_instance = int(exp.get("runs_per_instance", 10))

    weight_low = int(exp.get("weight_low", 1))
    weight_high = int(exp.get("weight_high", 50))
    value_low = int(exp.get("value_low", 1))
    value_high = int(exp.get("value_high", 100))
    capacity_ratio = float(exp.get("capacity_ratio", 0.5))

    budget = int(exp.get("budget_evals", 50000))
    penalty_lambda = float(exp.get("penalty_lambda", 1000.0))
    feasible_only = bool(exp.get("feasible_only", True))

    run_csv = exp.get("run_csv", "runs_knapsack.csv")
    trace_csv = exp.get("trace_csv", "trace_knapsack.csv")
    code_version = exp.get("code_version", "")

    logger = RunLogger(run_csv=run_csv, trace_csv=trace_csv)

    # Build solver variants
    variants = []  # (algo_name, tag, solver)
    for algo in algos:
        base_params = cfg.get(algo, {}) if isinstance(cfg.get(algo, {}), dict) else {}
        sweep_params = sweeps.get(algo, {}) if isinstance(sweeps.get(algo, {}), dict) else {}

        for sparams in _grid(sweep_params):
            params = dict(base_params)
            params.update(sparams)
            tag = _tag(sparams)

            if algo == "SA_KP":
                # ~1 eval per iteration => max_iter ~ budget
                solver = SimulatedAnnealingKnapsack(
                    T0=float(params.get("T0", 10.0)),
                    Tmin=float(params.get("Tmin", 1e-3)),
                    steps_per_temp=int(params.get("steps_per_temp", 2000)),
                    alpha=float(params.get("alpha", 0.99)),
                    max_iter=budget - 1,
                    trace_every=int(params.get("trace_every", 200)),
                    feasible_only=feasible_only,
                )

            elif algo == "GA_KP":
                pop = int(params.get("population_size", 100))
                gens = max(1, (budget - pop) // pop)  # evals ≈ pop + gens*pop
                solver = GeneticAlgorithmKnapsack(
                    population_size=pop,
                    generations=gens,
                    crossover_rate=float(params.get("crossover_rate", 0.9)),
                    mutation_rate=float(params.get("mutation_rate", 0.02)),
                    tournament_k=int(params.get("tournament_k", 10)),
                    trace_every=int(params.get("trace_every", 10)),
                    penalty_lambda=penalty_lambda,
                    feasible_only=feasible_only,
                )

            elif algo == "HC_KP":
                solver = HC_Knapsack(
                    max_iter=budget - 1,
                    mode=str(params.get("mode", "first")),
                    trace_every=int(params.get("trace_every", 200)),
                    flip_k=int(params.get("flip_k", 1)),
                    feasible_only=feasible_only,
                )

            elif algo == "TLBO_KP":
                # Warning: TLBO evals scale ~ pop + 2*pop*iters.
                # We'll keep iters from YAML (for sensitivity), but fairness should use best@budget in analysis.
                solver = TLBO_Knapsack(
                    pop_size=int(params.get("pop_size", 50)),
                    iters=int(params.get("iters", 500)),
                    trace_every=int(params.get("trace_every", 10)),
                    feasible_only=feasible_only,
                )

            elif algo == "ABC_KP":
                sn = int(params.get("sn", 50))
                # Roughly 2*sn evaluations per iter (employed + onlooker); adjust to hit budget
                iters = max(1, (budget - sn) // (2 * sn + sn * (2*sn / max(50000,1))))
                solver = ABC_Knapsack(
                    food_sources=sn,
                    iters=iters,
                    limit=int(params.get("limit", 50)),
                    trace_every=int(params.get("trace_every", 10)),
                    feasible_only=feasible_only,
                )
            else:
                continue

            variants.append((algo, tag, solver))

    # Run experiments
    for n_items in n_items_list:
        for inst in range(instances):
            problem = KnapsackProblem.random_instance(
                int(n_items),
                instance_seed=inst,
                weight_range=(weight_low, weight_high),
                value_range=(value_low, value_high),
                capacity_ratio=capacity_ratio,
            )

            for run in range(runs_per_instance):
                for algo_name, tag, solver in variants:
                    run_id = f"{algo_name}[{tag}]_n{n_items}_inst{inst}_run{run}"
                    solver.solve(
                        problem,
                        init_solution=None,
                        logger=logger,
                        experiment_id=exp_id,
                        run_id=run_id,
                        instance_seed=inst,
                        seed_algo=run,
                        code_version=code_version,
                    )

    print("Done.")
    print(f"Run log:   {run_csv}")
    print(f"Trace log: {trace_csv}")

if __name__ == "__main__":
    main()