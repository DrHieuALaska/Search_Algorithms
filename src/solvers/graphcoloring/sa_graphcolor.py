from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from core.base import RunResult
from core.logging import RunLogger
from core.rng import make_rng, combine_seeds
from problems.graph_coloring import GraphColoringProblem


class SimulatedAnnealingGraphColoring:
    """SA for k-coloring (minimize number of conflicting edges).

    evals_cost semantics
    --------------------
    One evals_cost unit = one delta_recolor() call = O(deg(v)) work.
    Consistent with HC_GC.  Normalized via eval_norm_gc (factor = 1/n).
    """

    name = "SA_GC"

    def __init__(
        self,
        *,
        T0: float          = 5.0,
        Tmin: float        = 1e-3,
        alpha: float       = 0.99,
        max_iter: int      = 50_000,
        trace_every: int   = 200,
    ):
        if not (T0 > 0 and Tmin > 0):
            raise ValueError("T0 and Tmin must be > 0")
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0,1)")
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if trace_every <= 0:
            raise ValueError("trace_every must be > 0")
        self.T0          = float(T0)
        self.Tmin        = float(Tmin)
        self.alpha       = float(alpha)
        self.max_iter    = int(max_iter)
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
            init_rng      = make_rng(combine_seeds(instance_seed, seed_algo))
            init_solution = problem.random_solution(init_rng)

        colors       = np.array(init_solution, copy=True, dtype=np.int32)
        init_cost    = problem.evaluate(colors)   # for logging only, not counted
        evals_cost   = 0

        current_cost = int(init_cost)
        best_cost    = int(init_cost)
        best_sol     = colors.copy()

        iters               = 0
        iter_best_found     = 0
        evals_best_found    = 0
        time_best_found_sec = 0.0

        trace_iter:  List[int]   = []
        trace_evals: List[int]   = []
        trace_time:  List[float] = []
        trace_temp:  List[float] = []
        trace_best:  List[float] = []
        trace_curr:  List[float] = []

        t_start = time.perf_counter()

        def elapsed() -> float:
            return time.perf_counter() - t_start

        def checkpoint(temp: float):
            trace_iter.append(iters)
            trace_evals.append(evals_cost)
            trace_time.append(elapsed())
            trace_temp.append(temp)
            trace_best.append(best_cost)
            trace_curr.append(current_cost)
            if logger is not None:
                logger.log_trace({
                    "experiment_id": experiment_id,
                    "run_id":        run_id,
                    "iter":          iters,
                    "evals_cost":    evals_cost,
                    "time_sec":      trace_time[-1],
                    "temp":          temp,
                    "best_cost":     best_cost,
                    "current_cost":  current_cost,
                })

        temp = self.T0
        checkpoint(temp)

        n = problem.n
        k = problem.k

        while temp > self.Tmin and iters < self.max_iter and best_cost > 0:
            v   = int(rng.integers(0, n))
            old = int(colors[v])
            new = int(rng.integers(0, k - 1))
            if new >= old:
                new += 1

            delta       = problem.delta_recolor(colors, v, new)
            evals_cost += 1

            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                colors[v]     = new
                current_cost += int(delta)
                if current_cost < best_cost:
                    best_cost           = int(current_cost)
                    best_sol            = colors.copy()
                    # BUG FIX: iters is incremented AFTER this block, so
                    # iter_best_found should record iters + 1 (the upcoming value).
                    # Use iters + 1 consistently (1-indexed iteration number).
                    iter_best_found     = iters + 1
                    evals_best_found    = evals_cost
                    time_best_found_sec = elapsed()

            iters += 1
            if iters % self.trace_every == 0:
                checkpoint(temp)

            temp *= self.alpha

        checkpoint(temp)

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
                "init_temp_T0":     self.T0,
                "min_temp_Tmin":    self.Tmin,
                "alpha":            self.alpha,
                "steps_per_temp":   "",
                "max_iter":         self.max_iter,
                "neighbor_operator": "recolor-1",
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
            "temp":         np.array(trace_temp,  dtype=np.float64),
            "best_cost":    np.array(trace_best,  dtype=np.float64),
            "current_cost": np.array(trace_curr,  dtype=np.float64),
        }
        return result, trace
