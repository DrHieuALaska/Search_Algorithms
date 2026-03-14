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



from solvers.tsp.sa_tsp import SimulatedAnnealingTSP
from solvers.tsp.tlbo_tsp import TLBO_TSP
from solvers.tsp.ga_tsp import GeneticAlgorithmTSP
from solvers.tsp.hc_tsp import HC_TSP
from solvers.tsp.aco_tsp import ACO_TSP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TSP experiments from YAML config")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., config_tsp_param.yaml)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    exp = cfg.get("experiment", {})
    algos = cfg.get("algos", ["SA"])

    # experiment settings
    exp_id = exp.get("exp_id", "exp_001")
    n_list = exp.get("n_list", [5, 15, 30])
    instances = int(exp.get("instances", 10))
    runs_per_instance = int(exp.get("runs_per_instance", 10))
    scale = float(exp.get("scale", 100.0))

    run_csv = exp.get("run_csv", "runs.csv")
    trace_csv = exp.get("trace_csv", "trace.csv")
    code_version = exp.get("code_version", "")

    logger = RunLogger(run_csv=run_csv, trace_csv=trace_csv)

    solvers = {}

    if "SA" in algos:
        p = cfg.get("SA", {})
        solvers["SA"] = SimulatedAnnealingTSP(
            T0=float(p.get("T0", 100.0)),
            Tmin=float(p.get("Tmin", 1e-3)),
            alpha=float(p.get("alpha", 0.99)),
            steps_per_temp=int(p.get("steps_per_temp", 2000)),
            max_iter=int(p.get("max_iter", 50000)),
            trace_every=int(p.get("trace_every", 200)),
        )

    if "TLBO" in algos:
        p = cfg.get("TLBO", {})
        solvers["TLBO"] = TLBO_TSP(
            pop_size=int(p.get("pop_size", 50)),
            iters=int(p.get("iters", 500)),
            max_evals=int(p.get("max_evals", 50_000)),
            trace_every=int(p.get("trace_every", 10)),
            move_frac_max=float(p.get("move_frac_max", 1.0))
        )

    if "GA" in algos:
        p = cfg.get("GA", {})
        ga_pop = int(p.get("pop", p.get("population_size", 100)))
        # generation may be a formula string like "(50k - pop)/pop"
        gen_raw = p.get("generation", p.get("generations", 400))
        try:
            ga_gen = int(gen_raw)
        except (ValueError, TypeError):
            ga_gen = (50_000 - ga_pop) // ga_pop
        solvers["GA"] = GeneticAlgorithmTSP(
            population_size=ga_pop,
            crossover_rate=float(p.get("crossover", p.get("crossover_rate", 0.8))),
            generations=ga_gen,
            mutation_rate=float(p.get("mutation_rate", 0.02)),
            tournament_k=int(p.get("tournament_k", 5)),
            trace_every=int(p.get("trace_every", 10)),
        )

    if "HC" in algos:
        p = cfg.get("HC", {})
        solvers["HC"] = HC_TSP(
            max_iter=int(p.get("max_iter", 50000)),
            mode=str(p.get("mode", "first")),
            trace_every=int(p.get("trace_every", 200)),
        )

    if "ACO" in algos:
        p = cfg.get("ACO", {})
        solvers["ACO"] = ACO_TSP(
            n_ants=int(p.get("n_ants", 30)),
            iters=int(p.get("iters", 1667)),
            alpha=float(p.get("alpha", 1.0)),
            beta=float(p.get("beta", 3.0)),
            rho=float(p.get("rho", 0.1)),
            Q=float(p.get("Q", 100.0)),
            trace_every=int(p.get("trace_every", 10)),
            deposit_best_only=bool(p.get("deposit_best_only", False)),
        )

    for n in n_list:
        for inst in range(instances):
            problem = TSPProblem(n_cities=int(n), instance_seed=inst, scale=scale)

            for run in range(runs_per_instance):
                for algo_name, solver in solvers.items():
                    run_id = f"{algo_name}_n{n}_inst{inst}_run{run}"
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
                    )

    print("Done.")
    print(f"Run log:   {run_csv}")
    print(f"Trace log: {trace_csv}")


if __name__ == "__main__":
    main()