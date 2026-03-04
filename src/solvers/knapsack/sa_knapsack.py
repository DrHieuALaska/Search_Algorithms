from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from core.base import RunResult
from core.logging import RunLogger
from core.rng import make_rng, combine_seeds
from problems.knapsack import KnapsackProblem


class SimulatedAnnealingKnapsack:
    """Simulated Annealing for 0/1 Knapsack (binary vector)."""

    name = "SA_KP"

    def __init__(
        self,
        *,
        T0: float = 10.0,
        Tmin: float = 1e-3,
        alpha: float = 0.99,
        steps_per_temp: int = 2000,
        max_iter: Optional[int] = None,
        trace_every: int = 200,
        feasible_only: bool = True,
    ):
        if not (T0 > 0 and Tmin > 0):
            raise ValueError("T0 and Tmin must be > 0")
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0,1)")
        if steps_per_temp <= 0:
            raise ValueError("steps_per_temp must be > 0")
        if trace_every <= 0:
            raise ValueError("trace_every must be > 0")
        self.T0 = float(T0)
        self.Tmin = float(Tmin)
        self.alpha = float(alpha)
        self.steps_per_temp = int(steps_per_temp)
        self.max_iter = max_iter
        self.trace_every = int(trace_every)
        self.feasible_only = bool(feasible_only)

    @staticmethod
    def _flip_one(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        y = x.copy()
        i = int(rng.integers(0, y.shape[0]))
        y[i] = 1 - y[i]
        return y

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

        # init solution reproducible per (instance_seed, seed_algo)
        if init_solution is None:
            init_rng = make_rng(combine_seeds(instance_seed, seed_algo))
            init_solution = problem.random_solution(init_rng, feasible=self.feasible_only)

        x0 = np.asarray(init_solution, dtype=np.int8)
        x = x0.copy()
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
        trace_temp: List[float] = []
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

        def checkpoint(temp: float):
            curr_value, curr_weight = problem.value_weight(x)
            best_value, best_weight = problem.value_weight(best_x)

            trace_iter.append(iters)
            trace_evals.append(evals_cost)
            trace_time.append(elapsed())
            trace_temp.append(temp)
            trace_best.append(best_cost)
            trace_curr.append(current_cost)
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
                    "temp": temp,
                    "best_cost": best_cost,
                    "current_cost": current_cost,
                    "best_value": best_value,
                    "current_value": curr_value,
                    "best_weight": best_weight,
                    "current_weight": curr_weight,
                    "is_feasible_best": trace_feas[-1],
                })

        temp = self.T0
        checkpoint(temp)

        while temp > self.Tmin:
            for _ in range(self.steps_per_temp):
                if self.max_iter is not None and iters >= self.max_iter:
                    temp = self.Tmin
                    break

                cand = self._flip_one(x, rng)
                if self.feasible_only:
                    cand = problem.repair(cand, rng)

                cand_cost = float(problem.evaluate(cand))
                evals_cost += 1
                delta = cand_cost - current_cost

                if delta <= 0.0 or rng.random() < math.exp(-delta / temp):
                    x = cand
                    current_cost = cand_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_x = x.copy()
                        iter_best_found = iters + 1
                        evals_best_found = evals_cost
                        time_best_found_sec = elapsed()

                iters += 1
                if iters % self.trace_every == 0:
                    checkpoint(temp)

            temp *= self.alpha

        checkpoint(temp)

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

                "init_temp_T0": self.T0,
                "min_temp_Tmin": self.Tmin,
                "alpha": self.alpha,
                "steps_per_temp": self.steps_per_temp,
                "max_iter": "" if self.max_iter is None else self.max_iter,
                "neighbor_operator": "flip1+repair" if self.feasible_only else "flip1+penalty",

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
            "temp": np.array(trace_temp, dtype=np.float64),
            "best_cost": np.array(trace_best, dtype=np.float64),
            "current_cost": np.array(trace_curr, dtype=np.float64),
            "best_value": np.array(trace_best_value, dtype=np.float64),
            "current_value": np.array(trace_curr_value, dtype=np.float64),
            "best_weight": np.array(trace_best_weight, dtype=np.float64),
            "current_weight": np.array(trace_curr_weight, dtype=np.float64),
            "is_feasible_best": np.array(trace_feas, dtype=np.int8),
        }
        return result, trace
