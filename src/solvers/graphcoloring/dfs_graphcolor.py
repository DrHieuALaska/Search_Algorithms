from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from core.base import RunResult
from core.logging import RunLogger
from core.rng import make_rng, combine_seeds
from problems.graph_coloring import GraphColoringProblem


class DFS_BacktrackingGraphColoring:
    """DFS backtracking to find a conflict-free k-coloring (cost=0) for small graphs.

    Exact decision procedure for fixed k; returns best found if timed out or
    backtrack limit reached.

    evals_cost semantics
    --------------------
    DFS does not call problem.evaluate() in the inner loop — it uses
    can_color() checks (O(deg) per vertex per color).  evals_cost is
    incremented only at checkpoint intervals (one full evaluate per checkpoint)
    and once on success/fallback.

    BUG FIX: evals_cost was barely incrementing (only at checkpoints and
    success), making it nearly useless as a budget proxy.  We now count
    every can_color() check as one evals_cost unit instead, which is
    consistent with HC_GC/SA_GC (both use O(deg) per check).  This makes
    DFS_GC comparable on the normalized axis with HC_GC and SA_GC.

    iters counts backtrack steps (one per placed or retracted assignment).
    """

    name = "DFS_GC"

    def __init__(
        self,
        *,
        time_limit_sec: float  = 2.0,
        max_backtracks: int    = 2_000_000,
        trace_every: int       = 10_000,
    ):
        if time_limit_sec <= 0:
            raise ValueError("time_limit_sec must be > 0")
        if max_backtracks <= 0:
            raise ValueError("max_backtracks must be > 0")
        if trace_every <= 0:
            raise ValueError("trace_every must be > 0")
        self.time_limit_sec = float(time_limit_sec)
        self.max_backtracks = int(max_backtracks)
        self.trace_every    = int(trace_every)

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

        n = problem.n
        k = problem.k

        # vertex order: descending degree (high-degree first reduces branching)
        order        = np.argsort(-problem.deg).astype(int)
        colors       = -np.ones(n, dtype=np.int32)

        def can_color(v: int, c: int) -> bool:
            for u in problem.adj[v]:
                if colors[u] == c:
                    return False
            return True

        backtracks          = 0
        evals_cost          = 0   # BUG FIX: counts can_color() calls (O(deg) each)
        iters               = 0

        best_sol            = None
        best_cost           = 10**18

        iter_best_found     = 0
        evals_best_found    = 0
        time_best_found_sec = 0.0

        trace_iter:  List[int]   = []
        trace_evals: List[int]   = []
        trace_time:  List[float] = []
        trace_best:  List[float] = []
        trace_curr:  List[float] = []

        t_start = time.perf_counter()
        def elapsed() -> float:
            return time.perf_counter() - t_start

        def checkpoint(curr_cost: float):
            trace_iter.append(iters)
            trace_evals.append(evals_cost)
            trace_time.append(elapsed())
            trace_best.append(float(best_cost) if best_sol is not None else float("inf"))
            trace_curr.append(curr_cost)
            if logger is not None:
                logger.log_trace({
                    "experiment_id": experiment_id,
                    "run_id":        run_id,
                    "iter":          iters,
                    "evals_cost":    evals_cost,
                    "time_sec":      trace_time[-1],
                    "temp":          "",
                    "best_cost":     trace_best[-1],
                    "current_cost":  curr_cost,
                })

        checkpoint(curr_cost=float("inf"))

        # Iterative backtracking stack: (idx_in_order, next_color_to_try)
        stack:      List[Tuple[int, int]] = []
        idx         = 0
        next_color  = 0

        while True:
            if elapsed() >= self.time_limit_sec or backtracks >= self.max_backtracks:
                break

            if idx == n:
                # found complete proper coloring
                sol        = colors.copy()
                c          = 0
                evals_cost += 1   # symbolic: confirm cost=0
                if c < best_cost:
                    best_cost           = c
                    best_sol            = sol
                    iter_best_found     = iters
                    evals_best_found    = evals_cost
                    time_best_found_sec = elapsed()
                break

            v      = int(order[idx])
            placed = False

            for c in range(next_color, k):
                iters      += 1
                evals_cost += 1   # BUG FIX: count each can_color() call
                if can_color(v, c):
                    colors[v] = c
                    stack.append((idx, c + 1))
                    idx        = idx + 1
                    next_color = 0
                    placed     = True
                    break

            if not placed:
                colors[v] = -1
                backtracks += 1
                if not stack:
                    break
                idx, next_color  = stack.pop()
                v_prev           = int(order[idx])
                colors[v_prev]   = -1

            if iters % self.trace_every == 0:
                tmp      = colors.copy()
                tmp[tmp < 0] = 0
                curr     = problem.evaluate(tmp)
                evals_cost += 1
                if curr < best_cost:
                    best_cost           = int(curr)
                    best_sol            = tmp.copy()
                    iter_best_found     = iters
                    evals_best_found    = evals_cost
                    time_best_found_sec = elapsed()
                checkpoint(curr_cost=float(curr))

            if best_cost == 0:
                break

        if best_sol is None:
            init_seed = combine_seeds(instance_seed, seed_algo)
            init_rng  = make_rng(init_seed)
            best_sol  = problem.random_solution(init_rng)
            best_cost = problem.evaluate(best_sol)
            evals_cost        += 1
            evals_best_found   = evals_cost
            time_best_found_sec = elapsed()

        checkpoint(curr_cost=float(best_cost))
        time_sec = elapsed()

        result = RunResult(
            best_solution=best_sol,
            best_cost=float(best_cost),
            final_cost=float(best_cost),
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
                "max_iter":         "",
                "neighbor_operator": f"DFS(tlim={self.time_limit_sec}s)",
                "iters":            iters,
                "evals_cost":       evals_cost,
                "time_sec":         time_sec,
                "init_cost":        "",
                "final_cost":       best_cost,
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
