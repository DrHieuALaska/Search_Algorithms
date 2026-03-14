from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from core.base import RunResult
from core.logging import RunLogger
from core.rng import make_rng, combine_seeds
from problems.knapsack import KnapsackProblem


class TLBO_Knapsack:
    """
    TLBO for 0/1 Knapsack (binary vector) - simple discrete adaptation.

    Teacher phase:
      - For each learner, adopt teacher bits on a random subset of positions where they differ.

    Learner phase:
      - For each learner i with partner j, if j is better (lower cost), move towards j similarly.

    Feasibility:
      - If feasible_only=True, repair after each update.
    """

    name = "TLBO_KP"

    def __init__(
        self,
        *,
        pop_size: int = 50,
        iters: int = 500,
        trace_every: int = 10,
        feasible_only: bool = True,
        move_frac_max: float = 0.4,   # probability to copy a differing bit
    ):
        if pop_size <= 2:
            raise ValueError("pop_size must be > 2")
        if iters <= 0:
            raise ValueError("iters must be > 0")
        if trace_every <= 0:
            raise ValueError("trace_every must be > 0")
        if not (0.0 < move_frac_max <= 1.0):
            raise ValueError("move_frac_max must be in (0,1]")
        self.pop_size = int(pop_size)
        self.iters = int(iters)
        self.trace_every = int(trace_every)
        self.feasible_only = bool(feasible_only)
        self.move_frac_max = float(move_frac_max)

    def _move_towards(self, x: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        y = x.copy()
        diff = (y != target)
        if not np.any(diff):
            return y
        mask = diff & (rng.random(y.shape[0]) < self.move_frac_max)
        y[mask] = target[mask]
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
        init_rng = make_rng(combine_seeds(instance_seed, seed_algo))

        pop = init_rng.integers(0, 2, size=(self.pop_size, problem.n), dtype=np.int8)
        if self.feasible_only:
            for i in range(self.pop_size):
                pop[i] = problem.repair(pop[i], init_rng)

        if init_solution is not None:
            x0 = np.asarray(init_solution, dtype=np.int8)
            pop[0] = problem.repair(x0, init_rng) if self.feasible_only else x0

        costs = np.array([problem.evaluate(pop[i]) for i in range(self.pop_size)], dtype=float)
        evals_cost = int(self.pop_size)

        best_idx = int(np.argmin(costs))
        best_x = pop[best_idx].copy()
        best_cost = float(costs[best_idx])
        init_cost = best_cost

        iter_best_found = 0
        evals_best_found = evals_cost
        time_best_found_sec = 0.0

        trace_iter: List[int] = []
        trace_evals: List[int] = []
        trace_time: List[float] = []
        trace_best: List[float] = []
        trace_best_value: List[float] = []
        trace_best_weight: List[float] = []

        t_start = time.perf_counter()

        def elapsed() -> float:
            return time.perf_counter() - t_start

        def checkpoint(it: int):
            bv, bw = problem.value_weight(best_x)
            trace_iter.append(it)
            trace_evals.append(evals_cost)
            trace_time.append(elapsed())
            trace_best.append(best_cost)
            trace_best_value.append(bv)
            trace_best_weight.append(bw)

            if logger is not None:
                logger.log_trace({
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "iter": it,
                    "evals_cost": evals_cost,
                    "time_sec": trace_time[-1],
                    "temp": "",
                    "best_cost": best_cost,
                    "current_cost": float(np.min(costs)),
                    "best_value": bv,
                    "current_value": "",
                    "best_weight": bw,
                    "current_weight": "",
                    "is_feasible_best": 1 if bw <= problem.capacity + 1e-9 else 0,
                })

        checkpoint(0)

        for it in range(1, self.iters + 1):
            teacher_idx = int(np.argmin(costs))
            teacher = pop[teacher_idx]

            # Teacher phase
            for i in range(self.pop_size):
                if i == teacher_idx:
                    continue
                cand = self._move_towards(pop[i], teacher, rng)
                if self.feasible_only:
                    cand = problem.repair(cand, rng)
                cand_cost = float(problem.evaluate(cand))
                evals_cost += 1
                if cand_cost < float(costs[i]):
                    pop[i] = cand
                    costs[i] = cand_cost
                    if cand_cost < best_cost:
                        best_cost = cand_cost
                        best_x = cand.copy()
                        iter_best_found = it
                        evals_best_found = evals_cost
                        time_best_found_sec = elapsed()

            # Learner phase
            for i in range(self.pop_size):
                j = int(rng.integers(0, self.pop_size - 1))
                if j >= i:
                    j += 1
                if costs[j] < costs[i]:
                    cand = self._move_towards(pop[i], pop[j], rng)
                    if self.feasible_only:
                        cand = problem.repair(cand, rng)
                    cand_cost = float(problem.evaluate(cand))
                    evals_cost += 1
                    if cand_cost < float(costs[i]):
                        pop[i] = cand
                        costs[i] = cand_cost
                        if cand_cost < best_cost:
                            best_cost = cand_cost
                            best_x = cand.copy()
                            iter_best_found = it
                            evals_best_found = evals_cost
                            time_best_found_sec = elapsed()

            if it % self.trace_every == 0:
                checkpoint(it)

        time_sec = elapsed()
        best_value, best_weight = problem.value_weight(best_x)

        final_best_idx = int(np.argmin(costs))
        final_cost = float(costs[final_best_idx])
        final_value, final_weight = problem.value_weight(pop[final_best_idx])

        result = RunResult(
            best_solution=best_x,
            best_cost=float(best_cost),
            final_cost=float(final_cost),
            evals_cost=int(evals_cost),
            iters=int(self.iters),
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
                "neighbor_operator": f"copydiff(p={self.move_frac_max})+repair" if self.feasible_only else f"copydiff(p={self.move_frac_max})+penalty",

                "iters": self.iters,
                "evals_cost": evals_cost,
                "time_sec": time_sec,

                "init_cost": init_cost,
                "final_cost": final_cost,
                "best_cost": best_cost,

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
                "final_value": final_value,
                "final_weight": final_weight,
                "is_feasible_best": 1 if best_weight <= problem.capacity + 1e-9 else 0,
            })

        trace = {
            "iter": np.array(trace_iter, dtype=np.int64),
            "evals_cost": np.array(trace_evals, dtype=np.int64),
            "time_sec": np.array(trace_time, dtype=np.float64),
            "best_cost": np.array(trace_best, dtype=np.float64),
            "best_value": np.array(trace_best_value, dtype=np.float64),
            "best_weight": np.array(trace_best_weight, dtype=np.float64),
        }
        return result, trace
