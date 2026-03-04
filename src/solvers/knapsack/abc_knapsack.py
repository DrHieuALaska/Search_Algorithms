from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from core.base import RunResult
from core.logging import RunLogger
from core.rng import make_rng, combine_seeds
from problems.knapsack import KnapsackProblem


class ABC_Knapsack:
    """Artificial Bee Colony (binary) for 0/1 Knapsack."""

    name = "ABC_KP"

    def __init__(
        self,
        *,
        food_sources: int = 50,   # number of solutions
        iters: int = 500,
        limit: int = 50,          # abandonment limit
        trace_every: int = 10,
        feasible_only: bool = True,
    ):
        if food_sources <= 2:
            raise ValueError("food_sources must be > 2")
        if iters <= 0:
            raise ValueError("iters must be > 0")
        if limit <= 0:
            raise ValueError("limit must be > 0")
        if trace_every <= 0:
            raise ValueError("trace_every must be > 0")

        self.food_sources = int(food_sources)
        self.iters = int(iters)
        self.limit = int(limit)
        self.trace_every = int(trace_every)
        self.feasible_only = bool(feasible_only)

    @staticmethod
    def _neighbor_binary(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Make a neighbor by flipping one bit among positions where x differs from y (or random bit)."""
        n = x.shape[0]
        cand = x.copy()
        diff = np.flatnonzero(cand != y)
        if diff.size > 0:
            j = int(diff[rng.integers(0, diff.size)])
        else:
            j = int(rng.integers(0, n))
        cand[j] = 1 - cand[j]
        return cand

    def solve(
        self,
        problem: KnapsackProblem,
        init_solution: Optional[np.ndarray] = None,
        logger: Optional[RunLogger] = None,
        **meta: Any
    ) -> Tuple[RunResult, Dict[str, np.ndarray]]:
        experiment_id = str(meta.get("experiment_id", "exp_001"))
        run_id = str(meta.get("run_id", "run_001"))
        instance_seed = int(meta.get("instance_seed", 0))
        seed_algo = int(meta.get("seed_algo", 0))
        code_version = str(meta.get("code_version", ""))

        rng = make_rng(seed_algo)

        # init population deterministic per (instance_seed, seed_algo)
        init_rng = make_rng(combine_seeds(instance_seed, seed_algo))
        foods = np.empty((self.food_sources, problem.n), dtype=np.int8)
        for i in range(self.food_sources):
            foods[i] = problem.random_solution(init_rng, feasible=self.feasible_only)

        if init_solution is not None:
            foods[0] = np.asarray(init_solution, dtype=np.int8)
            if self.feasible_only:
                foods[0] = problem.repair(foods[0], rng)

        # evaluate
        costs = np.array([problem.evaluate(foods[i]) for i in range(self.food_sources)], dtype=float)
        evals_cost = int(self.food_sources)
        trials = np.zeros(self.food_sources, dtype=np.int32)

        best_idx = int(np.argmin(costs))
        best_x = foods[best_idx].copy()
        best_cost = float(costs[best_idx])

        init_cost = best_cost
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
        trace_curr_value: List[float] = []
        trace_best_weight: List[float] = []
        trace_curr_weight: List[float] = []
        trace_feas: List[int] = []

        t_start = time.perf_counter()

        def elapsed() -> float:
            return time.perf_counter() - t_start

        def is_feasible(w: float) -> int:
            return 1 if w <= problem.capacity + 1e-9 else 0

        def checkpoint():
            # current = best in population (not necessarily global best)
            cur_idx = int(np.argmin(costs))
            cur_x = foods[cur_idx]
            cur_cost = float(costs[cur_idx])

            curr_value, curr_weight = problem.value_weight(cur_x)
            best_value, best_weight = problem.value_weight(best_x)

            trace_iter.append(iters)
            trace_evals.append(evals_cost)
            trace_time.append(elapsed())
            trace_best.append(best_cost)
            trace_curr.append(cur_cost)
            trace_best_value.append(best_value)
            trace_curr_value.append(curr_value)
            trace_best_weight.append(best_weight)
            trace_curr_weight.append(curr_weight)
            trace_feas.append(is_feasible(best_weight))

            if logger is not None:
                logger.log_trace({
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "iter": iters,
                    "evals_cost": evals_cost,
                    "time_sec": trace_time[-1],
                    "temp": "",  # not applicable
                    "best_cost": best_cost,
                    "current_cost": cur_cost,
                    "best_value": best_value,
                    "current_value": curr_value,
                    "best_weight": best_weight,
                    "current_weight": curr_weight,
                    "is_feasible_best": is_feasible(best_weight),
                })

        checkpoint()

        for it in range(1, self.iters + 1):
            iters = it

            # quality for roulette selection (higher is better): q = -cost
            q = -costs
            q_shift = q - q.min() + 1e-12
            probs = q_shift / q_shift.sum()

            # --- Employed bees ---
            for i in range(self.food_sources):
                # pick k != i
                k = int(rng.integers(0, self.food_sources - 1))
                if k >= i:
                    k += 1

                cand = self._neighbor_binary(foods[i], foods[k], rng)
                if self.feasible_only:
                    cand = problem.repair(cand, rng)
                cand_cost = float(problem.evaluate(cand))
                evals_cost += 1

                if cand_cost < float(costs[i]):
                    foods[i] = cand
                    costs[i] = cand_cost
                    trials[i] = 0
                else:
                    trials[i] += 1

            # --- Onlooker bees ---
            for _ in range(self.food_sources):
                i = int(rng.choice(self.food_sources, p=probs))
                k = int(rng.integers(0, self.food_sources - 1))
                if k >= i:
                    k += 1

                cand = self._neighbor_binary(foods[i], foods[k], rng)
                if self.feasible_only:
                    cand = problem.repair(cand, rng)
                cand_cost = float(problem.evaluate(cand))
                evals_cost += 1

                if cand_cost < float(costs[i]):
                    foods[i] = cand
                    costs[i] = cand_cost
                    trials[i] = 0
                else:
                    trials[i] += 1

            # --- Scout bees ---
            for i in range(self.food_sources):
                if trials[i] >= self.limit:
                    foods[i] = problem.random_solution(rng, feasible=self.feasible_only)
                    if self.feasible_only:
                        foods[i] = problem.repair(foods[i], rng)
                    costs[i] = float(problem.evaluate(foods[i]))
                    evals_cost += 1
                    trials[i] = 0

            # global best update
            idx = int(np.argmin(costs))
            if float(costs[idx]) < best_cost:
                best_cost = float(costs[idx])
                best_x = foods[idx].copy()
                iter_best_found = iters
                evals_best_found = evals_cost
                time_best_found_sec = elapsed()

            if iters % self.trace_every == 0:
                checkpoint()

        checkpoint()

        time_sec = elapsed()
        best_value, best_weight = problem.value_weight(best_x)
        curr_idx = int(np.argmin(costs))
        curr_value, curr_weight = problem.value_weight(foods[curr_idx])

        result = RunResult(
            best_solution=best_x,
            best_cost=float(best_cost),
            final_cost=float(costs[curr_idx]),
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
                "max_iter": self.iters,
                "neighbor_operator": f"ABC(SN={self.food_sources},limit={self.limit})",

                "iters": iters,
                "evals_cost": evals_cost,
                "time_sec": time_sec,

                "init_cost": init_cost,
                "final_cost": float(costs[curr_idx]),
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
            "current_value": np.array(trace_curr_value, dtype=np.float64),
            "best_weight": np.array(trace_best_weight, dtype=np.float64),
            "current_weight": np.array(trace_curr_weight, dtype=np.float64),
            "is_feasible_best": np.array(trace_feas, dtype=np.int8),
        }
        return result, trace
