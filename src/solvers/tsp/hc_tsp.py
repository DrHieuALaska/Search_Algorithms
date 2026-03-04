from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from core.base import RunResult
from core.logging import RunLogger
from core.rng import make_rng, combine_seeds
from problems.tsp import TSPProblem


class HC_TSP:
    """Hill Climbing for TSP using 2-opt. Modes: first-improvement or best-improvement."""

    name = "HC"

    def __init__(
        self,
        *,
        max_iter: int = 50_000,
        mode: str = "first",
        trace_every: int = 200,
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

    @staticmethod
    def _apply_2opt_inplace(tour: np.ndarray, i: int, k: int) -> None:
        tour[i:k + 1] = tour[i:k + 1][::-1]

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
        **meta: Any,
    ) -> Tuple[RunResult, Dict[str, np.ndarray]]:
        experiment_id = str(meta.get("experiment_id", "exp_001"))
        run_id = str(meta.get("run_id", "run_001"))
        instance_seed = int(meta.get("instance_seed", 0))
        seed_algo = int(meta.get("seed_algo", 0))
        coord_scale = float(meta.get("coord_scale", 100.0))
        code_version = str(meta.get("code_version", ""))

        rng = make_rng(seed_algo)

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
        trace_best: List[float] = []
        trace_curr: List[float] = []

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
                    "run_id": run_id,
                    "iter": iters,
                    "evals_cost": evals_cost,
                    "time_sec": trace_time[-1],
                    "temp": "",  # not applicable
                    "best_cost": best_cost,
                    "current_cost": current_cost,
                })

        checkpoint()

        if self.mode == "first":
            # First-improvement: sample a random 2-opt each step, accept if improving
            while iters < self.max_iter:
                i, k = self._sample_2opt_move(problem.n, rng)
                delta = problem.delta_2opt(tour, i, k)
                evals_cost += 1
                iters += 1

                if delta < 0.0:
                    self._apply_2opt_inplace(tour, i, k)
                    current_cost += delta

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_tour = tour.copy()
                        iter_best_found = iters
                        evals_best_found = evals_cost
                        time_best_found_sec = elapsed()

                if iters % self.trace_every == 0:
                    checkpoint()

        else:
            # Best-improvement: scan all 2-opt neighbors, apply best improving move; repeat
            improved = True
            while improved and iters < self.max_iter:
                improved = False
                best_delta = 0.0
                best_i, best_k = -1, -1

                for i in range(problem.n - 1):
                    for k in range(i + 1, problem.n):
                        if i == 0 and k == problem.n - 1:
                            continue
                        delta = problem.delta_2opt(tour, i, k)
                        evals_cost += 1
                        iters += 1

                        if delta < best_delta:
                            best_delta = delta
                            best_i, best_k = i, k

                        if iters % self.trace_every == 0:
                            checkpoint()

                        if iters >= self.max_iter:
                            break
                    if iters >= self.max_iter:
                        break

                if best_delta < 0.0 and best_i >= 0:
                    improved = True
                    self._apply_2opt_inplace(tour, best_i, best_k)
                    current_cost += best_delta

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_tour = tour.copy()
                        iter_best_found = iters
                        evals_best_found = evals_cost
                        time_best_found_sec = elapsed()

        checkpoint()

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

                "init_temp_T0": "",
                "min_temp_Tmin": "",
                "alpha": "",
                "steps_per_temp": "",
                "max_iter": self.max_iter,
                "neighbor_operator": f"2-opt-{self.mode}",

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
            "best_cost": np.array(trace_best, dtype=np.float64),
            "current_cost": np.array(trace_curr, dtype=np.float64),
        }
        return result, trace
