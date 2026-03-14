from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from core.base import RunResult
from core.logging import RunLogger
from core.rng import make_rng, combine_seeds
from problems.tsp import TSPProblem


class SimulatedAnnealingTSP:
    """Fast SA for TSP using 2-opt with O(1) symmetric delta."""

    name = "SA"

    def __init__(
        self,
        *,
        T0: float = 100.0,
        Tmin: float = 1e-3,
        alpha: float = 0.99,
        steps_per_temp: int = 2000,
        max_iter: Optional[int] = None,
        trace_every: int = 200,
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

    @staticmethod
    def _apply_2opt_inplace(tour: np.ndarray, i: int, k: int) -> None:
        tour[i:k+1] = tour[i:k+1][::-1]

    @staticmethod
    def _sample_2opt_move(n: int, rng: np.random.Generator) -> Tuple[int, int]:
        while True:
            i = int(rng.integers(0, n - 1))
            k = int(rng.integers(i + 1, n))
            if not (i == 0 and k == n - 1):
                return i, k

    def solve(
        self,
        problem: TSPProblem,
        init_solution: Optional[np.ndarray] = None,
        logger: Optional[RunLogger] = None,
        **meta: Any
    ) -> Tuple[RunResult, Dict[str, np.ndarray]]:
        """
        Required meta keys (recommended):
          - experiment_id, run_id, instance_seed, seed_algo, coord_scale, code_version
        """
        experiment_id = str(meta.get("experiment_id", "exp_001"))
        run_id = str(meta.get("run_id", "run_001"))
        instance_seed = int(meta.get("instance_seed", 0))
        seed_algo = int(meta.get("seed_algo", 0))
        coord_scale = float(meta.get("coord_scale", 100.0))
        code_version = str(meta.get("code_version", ""))

        rng = make_rng(seed_algo)

        # init solution: use combined seed so all algorithms start from same tour for a given pairing
        if init_solution is None:
            init_seed = combine_seeds(instance_seed, seed_algo)
            init_rng = make_rng(init_seed)
            init_solution = problem.random_tour(problem.n, init_rng)

        tour = np.array(init_solution, copy=True)
        init_cost = problem.evaluate(tour)

        current_cost = init_cost
        best_tour = tour.copy()
        best_cost = current_cost

        evals_cost = 1
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
                    "run_id": run_id,
                    "iter": iters,
                    "evals_cost": evals_cost,
                    "time_sec": trace_time[-1],
                    "temp": temp,
                    "best_cost": best_cost,
                    "current_cost": current_cost,
                })

        temp = self.T0
        checkpoint(temp)

        while temp > self.Tmin:
            for _ in range(self.steps_per_temp):
                if self.max_iter is not None and iters >= self.max_iter:
                    temp = self.Tmin
                    break

                i, k = self._sample_2opt_move(problem.n, rng)
                delta = problem.delta_2opt(tour, i, k)
                evals_cost += 1

                if delta <= 0.0 or rng.random() < math.exp(-delta / temp):
                    self._apply_2opt_inplace(tour, i, k)
                    current_cost += delta

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_tour = tour.copy()
                        iter_best_found = iters + 1
                        evals_best_found = evals_cost
                        time_best_found_sec = elapsed()

                iters += 1
                if iters % self.trace_every == 0:
                    checkpoint(temp)

            temp *= self.alpha

        checkpoint(temp)

        time_sec = elapsed()
        result = RunResult(
            best_solution=best_tour,
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
            timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            logger.log_run({
                "experiment_id": experiment_id,
                "run_id": run_id,
                "timestamp_utc": timestamp_utc,
                "algorithm": self.name,
                "problem": "TSP",
                "n_cities": problem.n,
                "instance_seed": instance_seed,
                "coord_scale": coord_scale,
                "distance_type": "euclidean",
                "init_temp_T0": self.T0,
                "min_temp_Tmin": self.Tmin,
                "alpha": self.alpha,
                "steps_per_temp": self.steps_per_temp,
                "max_iter": "" if self.max_iter is None else self.max_iter,
                "neighbor_operator": "2-opt",
                "iters": iters,
                "evals_cost": evals_cost,
                "time_sec": time_sec,
                "init_cost": init_cost,
                "final_cost": current_cost,
                "best_cost": best_cost,
                "iter_best_found": iter_best_found,
                "evals_best_found": evals_best_found,
                "time_best_found_sec": time_best_found_sec,
                "seed_algo": seed_algo,
                "code_version": code_version,
            })

        trace = {
            "iter": np.array(trace_iter, dtype=np.int64),
            "evals_cost": np.array(trace_evals, dtype=np.int64),
            "time_sec": np.array(trace_time, dtype=np.float64),
            "temp": np.array(trace_temp, dtype=np.float64),
            "best_cost": np.array(trace_best, dtype=np.float64),
            "current_cost": np.array(trace_curr, dtype=np.float64),
        }
        return result, trace
