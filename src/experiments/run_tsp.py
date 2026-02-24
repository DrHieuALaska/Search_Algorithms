from __future__ import annotations

import os
import sys
import argparse

# Make src importable when running as a script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.logging import RunLogger
from problems.tsp import TSPProblem
from solvers.sa_tsp import SimulatedAnnealingTSP
from solvers.tlbo_tsp import TLBO_TSP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TSP experiments (SA/TLBO scaffold)")
    p.add_argument("--algos", type=str, nargs="+", default=["SA"], help="Algorithms to run: SA TLBO")
    p.add_argument("--n_list", type=int, nargs="+", default=[5, 15, 30, 50], help="List of city counts")
    p.add_argument("--instances", type=int, default=10, help="#instances per n (instance_seed = 0..instances-1)")
    p.add_argument("--runs_per_instance", type=int, default=10, help="#runs per instance (seed_algo = 0..runs-1)")

    p.add_argument("--scale", type=float, default=100.0, help="Coordinate scale for random cities")
    p.add_argument("--exp_id", type=str, default="exp_001", help="Experiment id")

    # TLBO params
    p.add_argument("--tlbo_pop", type=int, default=50, help="TLBO population size")
    p.add_argument("--tlbo_iters", type=int, default=500, help="TLBO iterations")
    p.add_argument("--tlbo_trace_every", type=int, default=10, help="TLBO trace checkpoint frequency")

    # SA params
    p.add_argument("--T0", type=float, default=100.0)
    p.add_argument("--Tmin", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.99)
    p.add_argument("--steps_per_temp", type=int, default=2000)
    p.add_argument("--max_iter", type=int, default=50000)
    p.add_argument("--trace_every", type=int, default=200)

    # logging
    p.add_argument("--run_csv", type=str, default="runs.csv", help="Run-level CSV path")
    p.add_argument("--trace_csv", type=str, default="trace.csv", help="Trace CSV path")
    p.add_argument("--code_version", type=str, default="", help="Git commit/tag (optional)")

    return p.parse_args()


def main():
    args = parse_args()

    logger = RunLogger(run_csv=args.run_csv, trace_csv=args.trace_csv)

    solvers = {}
if "SA" in args.algos:
    solvers["SA"] = SimulatedAnnealingTSP(
        T0=args.T0, Tmin=args.Tmin, alpha=args.alpha,
        steps_per_temp=args.steps_per_temp,
        max_iter=args.max_iter,
        trace_every=args.trace_every
    )
if "TLBO" in args.algos:
    solvers["TLBO"] = TLBO_TSP(
        pop_size=args.tlbo_pop,
        iters=args.tlbo_iters,
        trace_every=args.tlbo_trace_every
    )


    for n in args.n_list:
        for inst in range(args.instances):
            problem = TSPProblem(n_cities=n, instance_seed=inst, scale=args.scale)

            for run in range(args.runs_per_instance):
                for algo_name, solver in solvers.items():
                    run_id = f"{algo_name}_n{n}_inst{inst}_run{run}"
                    solver.solve(
                    problem,
                    init_solution=None,
                    logger=logger,
                    experiment_id=args.exp_id,
                    run_id=run_id,
                    instance_seed=inst,
                    seed_algo=run,
                    coord_scale=args.scale,
                    code_version=args.code_version,
                )

    print("Done.")
    print(f"Run log:   {args.run_csv}")
    print(f"Trace log: {args.trace_csv}")


if __name__ == "__main__":
    main()
