from __future__ import annotations

import os
import sys
import argparse
import itertools
import math
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.logging import RunLogger
from core.config import load_yaml
from problems.graph_coloring import GraphColoringProblem

from solvers.graphcoloring.sa_graphcolor import SimulatedAnnealingGraphColoring
from solvers.graphcoloring.ga_graphcolor import GeneticAlgorithmGraphColoring
from solvers.graphcoloring.hc_graphcolor import HillClimbingGraphColoring
from solvers.graphcoloring.aco_graphcolor import ACO_GraphColoring
from solvers.graphcoloring.dfs_graphcolor import DFS_BacktrackingGraphColoring


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Graph Coloring experiments from YAML config")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/config_graphcolor.yaml)")
    return p.parse_args()


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
        kk = str(k).replace("_", "")
        v_str = str(v).replace(".", "p")
        parts.append(f"{kk}{v_str}")
    return "-".join(parts)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    exp = cfg.get("experiment", {})
    algos = cfg.get("algos", ["SA_GC"])
    sweeps = cfg.get("sweeps", {})

    exp_id = exp.get("exp_id", "exp_gc_001")
    n_nodes_list = exp.get("n_nodes_list", [30])
    k = int(exp.get("n_colors", 4))
    edge_prob = float(exp.get("edge_prob", 0.2))
    ensure_connected = bool(exp.get("ensure_connected", False))

    instances = int(exp.get("instances", 10))
    runs_per_instance = int(exp.get("runs_per_instance", 10))
    budget = int(exp.get("budget_evals", 50000))

    run_csv = exp.get("run_csv", "runs_graphcolor.csv")
    trace_csv = exp.get("trace_csv", "trace_graphcolor.csv")
    code_version = exp.get("code_version", "")

    logger = RunLogger(run_csv=run_csv, trace_csv=trace_csv)

    # build solver variants (algo_name, tag, solver)
    variants = []
    for algo in algos:
        base_params = cfg.get(algo, {}) if isinstance(cfg.get(algo, {}), dict) else {}
        sweep_params = sweeps.get(algo, {}) if isinstance(sweeps.get(algo, {}), dict) else {}

        for sparams in _grid(sweep_params):
            params = dict(base_params)
            params.update(sparams)
            tag = _tag(sparams)

            if algo == "SA_GC":
                solver = SimulatedAnnealingGraphColoring(
                    T0=float(params.get("T0", 5.0)),
                    Tmin=float(params.get("Tmin", 1e-3)),
                    alpha=float(params.get("alpha", 0.99)),
                    max_iter=int(params.get("max_iter", budget - 1)),
                    trace_every=int(params.get("trace_every", 200)),
                )
            elif algo == "HC_GC":
                solver = HillClimbingGraphColoring(
                    max_iter=int(params.get("max_iter", budget - 1)),
                    mode=str(params.get("mode", "first")),
                    trace_every=int(params.get("trace_every", 200)),
                )
            elif algo == "GA_GC":
                pop = int(params.get("population_size", 100))
                gens = int(params.get("generations", max(1, (budget - pop) // pop)))
                solver = GeneticAlgorithmGraphColoring(
                    population_size=pop,
                    generations=gens,
                    crossover_rate=float(params.get("crossover_rate", 0.9)),
                    mutation_rate=float(params.get("mutation_rate", 0.02)),
                    tournament_k=int(params.get("tournament_k", 10)),
                    trace_every=int(params.get("trace_every", 10)),
                )
            elif algo == "ACO_GC":
                ants = int(params.get("n_ants", 30))
                iters = int(params.get("iters", max(1, math.ceil(budget / ants))))
                solver = ACO_GraphColoring(
                    n_ants=ants,
                    iters=iters,
                    alpha=float(params.get("alpha", 1.0)),
                    beta=float(params.get("beta", 2.0)),
                    rho=float(params.get("rho", 0.1)),
                    Q=float(params.get("Q", 1.0)),
                    trace_every=int(params.get("trace_every", 10)),
                    deposit_best_only=bool(params.get("deposit_best_only", True)),
                )
            elif algo == "DFS_GC":
                solver = DFS_BacktrackingGraphColoring(
                    time_limit_sec=float(params.get("time_limit_sec", 2.0)),
                    max_backtracks=int(params.get("max_backtracks", 2_000_000)),
                    trace_every=int(params.get("trace_every", 10000)),
                )
            else:
                continue

            variants.append((algo, tag, solver))

    # run experiments
    for n_nodes in n_nodes_list:
        n_nodes = int(n_nodes)
        for inst in range(instances):
            problem = GraphColoringProblem.random_instance(
                n_nodes=n_nodes,
                k=k,
                edge_prob=edge_prob,
                instance_seed=inst,
                ensure_connected=ensure_connected,
            )

            for run in range(runs_per_instance):
                for algo_name, tag, solver in variants:
                    run_id = f"{algo_name}[{tag}]_n{n_nodes}_inst{inst}_run{run}"
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
