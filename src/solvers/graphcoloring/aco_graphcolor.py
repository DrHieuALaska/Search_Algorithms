from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from core.base import RunResult
from core.logging import RunLogger
from core.rng import make_rng
from problems.graph_coloring import GraphColoringProblem


class ACO_GraphColoring:
    """ACO for k-coloring using pheromone tau[node, color].

    evals_cost semantics
    --------------------
    One evals_cost unit = one problem.evaluate() call = O(|E|) work.
    evals_cost += n_ants per iteration (one full evaluate per ant).
    Reference unit for GC eval_norm (factor = 1.0 for ACO_GC).

    BUG FIX: evals_best_found was left at 0 when the fallback random solution
    was used (iters==0 edge case).  Now correctly set to evals_cost after
    the fallback evaluate().
    """

    name = "ACO_GC"

    def __init__(
        self,
        *,
        n_ants: int              = 30,
        iters: int               = 1000,
        alpha: float             = 1.0,
        beta: float              = 2.0,
        rho: float               = 0.1,
        Q: float                 = 1.0,
        trace_every: int         = 10,
        deposit_best_only: bool  = True,
    ):
        if n_ants <= 0:
            raise ValueError("n_ants must be > 0")
        if iters <= 0:
            raise ValueError("iters must be > 0")
        if alpha < 0 or beta < 0:
            raise ValueError("alpha/beta must be >= 0")
        if not (0 < rho < 1):
            raise ValueError("rho must be in (0,1)")
        if Q <= 0:
            raise ValueError("Q must be > 0")
        if trace_every <= 0:
            raise ValueError("trace_every must be > 0")

        self.n_ants           = int(n_ants)
        self.iters            = int(iters)
        self.alpha            = float(alpha)
        self.beta             = float(beta)
        self.rho              = float(rho)
        self.Q                = float(Q)
        self.trace_every      = int(trace_every)
        self.deposit_best_only = bool(deposit_best_only)

    def _construct(
        self,
        problem: GraphColoringProblem,
        tau: np.ndarray,
        rng: np.random.Generator,
        order: np.ndarray,
    ) -> np.ndarray:
        n      = problem.n
        k      = problem.k
        colors = -np.ones(n, dtype=np.int32)

        for idx in range(n):
            v = int(order[idx])

            desir = np.empty(k, dtype=np.float64)
            for c in range(k):
                conf = 0
                for u in problem.adj[v]:
                    cu = int(colors[u])
                    if cu == c:
                        conf += 1
                eta      = 1.0 / (1.0 + conf)
                desir[c] = (tau[v, c] ** self.alpha) * (eta ** self.beta)

            s = desir.sum()
            if not np.isfinite(s) or s <= 0:
                chosen = int(rng.integers(0, k))
            else:
                p      = desir / s
                chosen = int(rng.choice(np.arange(k), p=p))
            colors[v] = chosen

        # fill any unassigned (shouldn't happen but guard anyway)
        mask = colors < 0
        if mask.any():
            colors[mask] = rng.integers(0, k, size=int(mask.sum()), dtype=np.int32)

        return colors

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

        rng   = make_rng(seed_algo)
        n     = problem.n
        k     = problem.k
        order = np.argsort(-problem.deg)
        tau   = np.full((n, k), 1.0, dtype=np.float64)

        best_sol  = None
        best_cost = 10**18

        evals_cost          = 0
        it                  = 0
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

        def checkpoint(curr_best: float):
            trace_iter.append(it)
            trace_evals.append(evals_cost)
            trace_time.append(elapsed())
            trace_best.append(float(best_cost) if best_cost < 10**18 else float("inf"))
            trace_curr.append(curr_best)
            if logger is not None:
                logger.log_trace({
                    "experiment_id": experiment_id,
                    "run_id":        run_id,
                    "iter":          it,
                    "evals_cost":    evals_cost,
                    "time_sec":      trace_time[-1],
                    "temp":          "",
                    "best_cost":     trace_best[-1],
                    "current_cost":  curr_best,
                })

        checkpoint(curr_best=float("inf"))

        for it_ in range(1, self.iters + 1):
            it   = it_
            sols  = np.empty((self.n_ants, n), dtype=np.int32)
            costs = np.empty(self.n_ants, dtype=np.int32)

            for a in range(self.n_ants):
                sol      = self._construct(problem, tau, rng, order)
                c        = problem.evaluate(sol)
                sols[a]  = sol
                costs[a] = c
            evals_cost += self.n_ants

            idx       = int(np.argmin(costs))
            iter_best = int(costs[idx])

            if iter_best < best_cost:
                best_cost           = int(iter_best)
                best_sol            = sols[idx].copy()
                iter_best_found     = it
                evals_best_found    = evals_cost
                time_best_found_sec = elapsed()

            # evaporate
            tau *= (1.0 - self.rho)

            # deposit
            use = [idx] if self.deposit_best_only else list(range(self.n_ants))
            for a in use:
                c     = int(costs[a])
                delta = self.Q / (1.0 + c)
                sol   = sols[a]
                for v in range(n):
                    tau[v, int(sol[v])] += delta

            if it % self.trace_every == 0:
                checkpoint(curr_best=float(iter_best))

            if best_cost == 0:
                break

        # BUG FIX: if no iterations ran (shouldn't happen but defensive),
        # evals_best_found must be set after the fallback evaluate().
        if best_sol is None:
            best_sol  = problem.random_solution(rng)
            best_cost = problem.evaluate(best_sol)
            evals_cost        += 1
            evals_best_found   = evals_cost   # ← was left at 0 before fix
            time_best_found_sec = elapsed()

        checkpoint(curr_best=float(best_cost))
        time_sec = elapsed()

        result = RunResult(
            best_solution=best_sol,
            best_cost=float(best_cost),
            final_cost=float(best_cost),
            evals_cost=int(evals_cost),
            iters=int(it),
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
                "max_iter":         self.iters,
                "neighbor_operator": f"tau[v,c](rho={self.rho},beta={self.beta})",
                "iters":            it,
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
