from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from core.base import RunResult
from core.logging import RunLogger
from core.rng import make_rng, combine_seeds
from problems.knapsack import KnapsackProblem


class HC_Knapsack:
    """Hill Climbing for 0/1 Knapsack using single-bit flips."""

    name = "HC_KP"

    def __init__(
        self,
        *,
        max_iter: int = 50_000,
        mode: str = "first",
        trace_every: int = 200,
        flip_k: int = 1,
        feasible_only: bool = True,
    ):
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if mode not in ("first", "best"):
            raise ValueError("mode must be 'first' or 'best'")
        if trace_every <= 0:
            raise ValueError("trace_every must be > 0")
        self.max_iter = int(max_iter)
        self.mode = str(mode)
        self.trace_every = int(trace_every)
        self.feasible_only = bool(feasible_only)

    @staticmethod
    def _flip(x: np.ndarray, i: int) -> np.ndarray:
        y = x.copy()
        y[i] = 1 - y[i]
        return y

    def solve(
        self,
        problem: KnapsackProblem,
        init_solution: Optional[np.ndarray] = None,
        logger: Optional[RunLogger] = None,
        **meta: Any,
    ) -> Tuple[RunResult, Dict[str, np.ndarray]]:
        experiment_id = str(meta.get("experiment_id", "exp_001"))
        run_id = str(meta.get("run_id", "run_001"))
        instance_seed = int(meta.get("instance_seed", 0))
        seed_algo = int(meta.get("seed_algo", 0))
        code_version = str(meta.get("code_version", ""))

        rng = make_rng(seed_algo)

        if init_solution is None:
            init_rng = make_rng(combine_seeds(instance_seed, seed_algo))
            init_solution = problem.random_solution(init_rng, feasible=self.feasible_only)

        x = np.asarray(init_solution, dtype=np.int8).copy()
        if self.feasible_only:
            x = problem.repair(x, rng)

        init_cost = float(problem.evaluate(x))
        current_cost = init_cost
        evals_cost = 1

        best_x = x.copy()
        best_cost = current_cost

        iters = 0
        iter_best_found = 0
        evals_best_found = evals_cost
        time_best_found_sec = 0.0

        trace_iter: List[int] = []
        trace_evals: List[int] = []
        trace_time: List[float] = []
        trace_best: List[float] = []
        trace_curr: List[float] = []
        trace_best_value: List[float] = []
        trace_best_weight: List[float] = []

        t_start = time.perf_counter()

        def elapsed() -> float:
            return time.perf_counter() - t_start

        def checkpoint():
            bv, bw = problem.value_weight(best_x)
            trace_iter.append(iters)
            trace_evals.append(evals_cost)
            trace_time.append(elapsed())
            trace_best.append(best_cost)
            trace_curr.append(current_cost)
            trace_best_value.append(bv)
            trace_best_weight.append(bw)

            if logger is not None:
                logger.log_trace({
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "iter": iters,
                    "evals_cost": evals_cost,
                    "time_sec": trace_time[-1],
                    "temp": "",
                    "best_cost": best_cost,
                    "current_cost": current_cost,
                    "best_value": bv,
                    "current_value": "",
                    "best_weight": bw,
                    "current_weight": "",
                    "is_feasible_best": 1 if bw <= problem.capacity + 1e-9 else 0,
                })

        checkpoint()

        n = problem.n

        if self.mode == "first":
            while iters < self.max_iter:
                i = int(rng.integers(0, n))
                cand = self._flip(x, i)
                if self.feasible_only:
                    cand = problem.repair(cand, rng)
                cand_cost = float(problem.evaluate(cand))
                evals_cost += 1
                iters += 1

                if cand_cost < current_cost:
                    x = cand
                    current_cost = cand_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_x = x.copy()
                        iter_best_found = iters
                        evals_best_found = evals_cost
                        time_best_found_sec = elapsed()

                if iters % self.trace_every == 0:
                    checkpoint()

        else:
            improved = True
            while improved and iters < self.max_iter:
                improved = False
                best_delta = 0.0
                best_cand = None

                base_cost = current_cost
                for i in range(n):
                    if iters >= self.max_iter:
                        break
                    cand = self._flip(x, i)
                    if self.feasible_only:
                        cand = problem.repair(cand, rng)
                    cand_cost = float(problem.evaluate(cand))
                    evals_cost += 1
                    iters += 1

                    delta = cand_cost - base_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_cand = cand

                    if iters % self.trace_every == 0:
                        checkpoint()

                if best_cand is not None and best_delta < 0.0:
                    improved = True
                    x = best_cand
                    current_cost = base_cost + best_delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_x = x.copy()
                        iter_best_found = iters
                        evals_best_found = evals_cost
                        time_best_found_sec = elapsed()

        checkpoint()

        time_sec = elapsed()
        best_value, best_weight = problem.value_weight(best_x)
        curr_value, curr_weight = problem.value_weight(x)

        result = RunResult(
            best_solution=best_x,
            best_cost=float(best_cost),
            final_cost=float(current_cost),
            evals_cost=int(evals_cost),
            iters=int(iters),
            time_sec=float(time_sec),
            iter_best_found=int(iter_best_found),
            evals_best_found=int(evals_best_found),
            time_best_found_sec=float(time_best_found_sec),
            meta={**dict(meta), "best_value": best_value, "best_weight": best_weight},
        )

        if logger is not None:
            timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            logger.log_run({
                "experiment_id": experiment_id,
                "run_id": run_id,
                "timestamp_utc": timestamp_utc,
                "algorithm": self.name,
                "problem": "KNAPSACK",
                "n_cities": "",
                "instance_seed": instance_seed,
                "coord_scale": "",
                "distance_type": "",

                "init_temp_T0": "",
                "min_temp_Tmin": "",
                "alpha": "",
                "steps_per_temp": "",
                "max_iter": self.max_iter,
                "neighbor_operator": f"flip1-{self.mode}+repair" if self.feasible_only else f"flip1-{self.mode}+penalty",

                "iters": iters,
                "evals_cost": evals_cost,
                "time_sec": time_sec,

                "init_cost": init_cost,
                "final_cost": float(current_cost),
                "best_cost": float(best_cost),

                "iter_best_found": iter_best_found,
                "evals_best_found": evals_best_found,
                "time_best_found_sec": time_best_found_sec,

                "seed_algo": seed_algo,
                "code_version": code_version,

                "n_items": problem.n,
                "capacity": problem.capacity,
                "capacity_ratio": "",
                "penalty_lambda": problem.penalty_lambda,
                "best_value": best_value,
                "best_weight": best_weight,
                "final_value": curr_value,
                "final_weight": curr_weight,
                "is_feasible_best": 1 if best_weight <= problem.capacity + 1e-9 else 0,
            })

        trace = {
            "iter": np.array(trace_iter, dtype=np.int64),
            "evals_cost": np.array(trace_evals, dtype=np.int64),
            "time_sec": np.array(trace_time, dtype=np.float64),
            "best_cost": np.array(trace_best, dtype=np.float64),
            "current_cost": np.array(trace_curr, dtype=np.float64),
            "best_value": np.array(trace_best_value, dtype=np.float64),
            "best_weight": np.array(trace_best_weight, dtype=np.float64),
        }
        return result, trace
