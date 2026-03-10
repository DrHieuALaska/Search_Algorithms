from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from core.base import RunResult
from core.logging import RunLogger
from core.rng import make_rng, combine_seeds
from problems.graph_coloring import GraphColoringProblem


class HillClimbingGraphColoring:
    """Hill Climbing for k-coloring using recolor moves.

    evals_cost semantics
    --------------------
    One evals_cost unit = one delta_recolor() call = O(deg(v)) work.
    This is consistent with SA_GC which uses the same move operator.
    GA_GC and ACO_GC use full evaluate() — their evals_cost is on a different
    raw scale, but normalized_evals makes all four comparable.

    mode='first': random recolor move; accept if delta < 0.
    mode='best':  sweep all (v, new_color) pairs per iteration and apply the
                  single best improving move found.  iters counts full sweeps,
                  NOT individual neighbor probes.
    """

    name = "HC_GC"

    def __init__(self, *, max_iter: int = 50_000, mode: str = "first", trace_every: int = 200):
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if mode not in ("first", "best"):
            raise ValueError("mode must be 'first' or 'best'")
        if trace_every <= 0:
            raise ValueError("trace_every must be > 0")
        self.max_iter   = int(max_iter)
        self.mode       = str(mode)
        self.trace_every = int(trace_every)

    def solve(
        self,
        problem: GraphColoringProblem,
        init_solution: Optional[np.ndarray] = None,
        logger: Optional[RunLogger] = None,
        **meta: Any
    ) -> Tuple[RunResult, Dict[str, np.ndarray]]:
        experiment_id = str(meta.get("experiment_id", "exp_gc_001"))
        run_id        = str(meta.get("run_id",        "run_001"))
        instance_seed = int(meta.get("instance_seed", 0))
        seed_algo     = int(meta.get("seed_algo",     0))
        code_version  = str(meta.get("code_version",  ""))

        rng = make_rng(seed_algo)
        if init_solution is None:
            init_rng       = make_rng(combine_seeds(instance_seed, seed_algo))
            init_solution  = problem.random_solution(init_rng)

        colors       = np.array(init_solution, copy=True, dtype=np.int32)
        # BUG FIX: do NOT count the initial evaluate() as evals_cost — HC_GC
        # only increments evals_cost for delta_recolor calls so that the
        # reference unit is consistent throughout (1 unit = 1 delta_recolor).
        init_cost    = problem.evaluate(colors)   # for logging only, not counted
        evals_cost   = 0

        current_cost = int(init_cost)
        best_cost    = int(init_cost)
        best_sol     = colors.copy()

        iters              = 0
        iter_best_found    = 0
        evals_best_found   = 0
        time_best_found_sec = 0.0

        trace_iter:  List[int]   = []
        trace_evals: List[int]   = []
        trace_time:  List[float] = []
        trace_best:  List[float] = []
        trace_curr:  List[float] = []

        t_start = time.perf_counter()
        def elapsed() -> float:
            return time.perf_counter() - t_start

        def checkpoint():
            trace_iter.append(iters)
            trace_evals.append(evals_cost)
            trace_time.append(elapsed())
            trace_best.append(best_cost)
            trace_curr.append(current_cost)
            if logger is not None:
                logger.log_trace({
                    "experiment_id": experiment_id,
                    "run_id":        run_id,
                    "iter":          iters,
                    "evals_cost":    evals_cost,
                    "time_sec":      trace_time[-1],
                    "temp":          "",
                    "best_cost":     best_cost,
                    "current_cost":  current_cost,
                })

        checkpoint()

        n = problem.n
        k = problem.k

        if self.mode == "first":
            while iters < self.max_iter and best_cost > 0:
                v   = int(rng.integers(0, n))
                old = int(colors[v])
                new = int(rng.integers(0, k - 1))
                if new >= old:
                    new += 1

                delta       = problem.delta_recolor(colors, v, new)
                evals_cost += 1
                iters      += 1

                if delta < 0:
                    colors[v]     = new
                    current_cost += int(delta)
                    if current_cost < best_cost:
                        best_cost           = int(current_cost)
                        best_sol            = colors.copy()
                        iter_best_found     = iters
                        evals_best_found    = evals_cost
                        time_best_found_sec = elapsed()

                if iters % self.trace_every == 0:
                    checkpoint()

        else:  # mode == "best"
            # BUG FIX: iters now counts full sweeps, not individual probes.
            # evals_cost still counts every delta_recolor call (n*(k-1) per sweep).
            improved = True
            while improved and iters < self.max_iter and best_cost > 0:
                improved   = False
                best_delta = 0
                best_v     = -1
                best_new   = -1

                for v in range(n):
                    old = int(colors[v])
                    for new in range(k):
                        if new == old:
                            continue
                        delta       = problem.delta_recolor(colors, v, new)
                        evals_cost += 1
                        if delta < best_delta:
                            best_delta = delta
                            best_v     = v
                            best_new   = new

                iters += 1   # one full sweep = one iteration

                if best_delta < 0 and best_v >= 0:
                    improved       = True
                    colors[best_v] = best_new
                    current_cost  += int(best_delta)
                    if current_cost < best_cost:
                        best_cost           = int(current_cost)
                        best_sol            = colors.copy()
                        iter_best_found     = iters
                        evals_best_found    = evals_cost
                        time_best_found_sec = elapsed()

                if iters % self.trace_every == 0:
                    checkpoint()

        checkpoint()

        time_sec = elapsed()
        result = RunResult(
            best_solution=best_sol,
            best_cost=float(best_cost),
            final_cost=float(current_cost),
            evals_cost=int(evals_cost),
            iters=int(iters),
            time_sec=float(time_sec),
            iter_best_found=int(iter_best_found),
            evals_best_found=int(evals_best_found),
            time_best_found_sec=float(time_best_found_sec),
            meta=dict(meta),
        )

        if logger is not None:
            ts_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            logger.log_run({
                "experiment_id":    experiment_id,
                "run_id":           run_id,
                "timestamp_utc":    ts_utc,
                "algorithm":        self.name,
                "problem":          "GRAPH_COLORING",
                "n_cities":         problem.n,
                "instance_seed":    instance_seed,
                "coord_scale":      "",
                "distance_type":    "graph",
                "init_temp_T0":     "",
                "min_temp_Tmin":    "",
                "alpha":            "",
                "steps_per_temp":   "",
                "max_iter":         self.max_iter,
                "neighbor_operator": f"recolor-1-{self.mode}",
                "iters":            iters,
                "evals_cost":       evals_cost,
                "time_sec":         time_sec,
                "init_cost":        init_cost,
                "final_cost":       current_cost,
                "best_cost":        best_cost,
                "iter_best_found":  iter_best_found,
                "evals_best_found": evals_best_found,
                "time_best_found_sec": time_best_found_sec,
                "seed_algo":        seed_algo,
                "code_version":     code_version,
            })

        trace = {
            "iter":         np.array(trace_iter,  dtype=np.int64),
            "evals_cost":   np.array(trace_evals, dtype=np.int64),
            "time_sec":     np.array(trace_time,  dtype=np.float64),
            "best_cost":    np.array(trace_best,  dtype=np.float64),
            "current_cost": np.array(trace_curr,  dtype=np.float64),
        }
        return result, trace
