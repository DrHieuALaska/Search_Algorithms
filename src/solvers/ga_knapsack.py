from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from core.base import RunResult
from core.logging import RunLogger
from core.rng import make_rng, combine_seeds
from problems.knapsack import KnapsackProblem


class GeneticAlgorithmKnapsack:
    """Binary GA for 0/1 Knapsack: tournament selection + 1-point crossover + bit-flip mutation."""

    name = "GA_KP"

    def __init__(
        self,
        *,
        population_size: int = 100,
        generations: int = 400,
        mutation_rate: float = 0.02,   # per-bit flip probability
        tournament_k: int = 10,
        trace_every: int = 10,
        feasible_only: bool = True,
    ):
        if population_size <= 2:
            raise ValueError("population_size must be > 2")
        if generations <= 0:
            raise ValueError("generations must be > 0")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0,1]")
        if tournament_k <= 0 or tournament_k > population_size:
            raise ValueError("tournament_k must be in [1, population_size]")
        if trace_every <= 0:
            raise ValueError("trace_every must be > 0")
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.mutation_rate = float(mutation_rate)
        self.tournament_k = int(tournament_k)
        self.trace_every = int(trace_every)
        self.feasible_only = bool(feasible_only)

    def _tournament(self, costs: np.ndarray, rng: np.random.Generator) -> int:
        idx = rng.choice(costs.shape[0], self.tournament_k, replace=False)
        return int(idx[np.argmin(costs[idx])])

    def _crossover_1pt(self, a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = a.shape[0]
        cut = int(rng.integers(1, n))
        child = np.empty(n, dtype=np.int8)
        child[:cut] = a[:cut]
        child[cut:] = b[cut:]
        return child

    def _mutate(self, x: np.ndarray, rng: np.random.Generator) -> None:
        if self.mutation_rate <= 0:
            return
        flip = rng.random(x.shape[0]) < self.mutation_rate
        x[flip] = 1 - x[flip]

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

        pop = init_rng.integers(0, 2, size=(self.population_size, problem.n), dtype=np.int8)
        if self.feasible_only:
            for i in range(self.population_size):
                pop[i] = problem.repair(pop[i], init_rng)

        if init_solution is not None:
            x0 = np.asarray(init_solution, dtype=np.int8)
            pop[0] = problem.repair(x0, init_rng) if self.feasible_only else x0

        # evaluate
        costs = np.array([problem.evaluate(pop[i]) for i in range(self.population_size)], dtype=float)
        evals_cost = int(self.population_size)

        best_idx = int(np.argmin(costs))
        best_x = pop[best_idx].copy()
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
        trace_best_weight: List[float] = []

        t_start = time.perf_counter()

        def elapsed() -> float:
            return time.perf_counter() - t_start

        def checkpoint():
            best_value, best_weight = problem.value_weight(best_x)
            trace_iter.append(iters)
            trace_evals.append(evals_cost)
            trace_time.append(elapsed())
            trace_best.append(best_cost)
            trace_curr.append(float(np.min(costs)))
            trace_best_value.append(best_value)
            trace_best_weight.append(best_weight)

            if logger is not None:
                logger.log_trace({
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "iter": iters,
                    "evals_cost": evals_cost,
                    "time_sec": trace_time[-1],
                    "temp": "",
                    "best_cost": best_cost,
                    "current_cost": trace_curr[-1],
                    "best_value": best_value,
                    "current_value": "",   # not tracked here
                    "best_weight": best_weight,
                    "current_weight": "",
                    "is_feasible_best": 1 if best_weight <= problem.capacity + 1e-9 else 0,
                })

        checkpoint()

        for gen in range(1, self.generations + 1):
            new_pop = np.empty_like(pop)

            # elitism: keep best
            elite_idx = int(np.argmin(costs))
            new_pop[0] = pop[elite_idx].copy()

            for i in range(1, self.population_size):
                p1 = pop[self._tournament(costs, rng)]
                p2 = pop[self._tournament(costs, rng)]
                child = self._crossover_1pt(p1, p2, rng)
                self._mutate(child, rng)
                if self.feasible_only:
                    child = problem.repair(child, rng)
                new_pop[i] = child

            pop = new_pop
            costs = np.array([problem.evaluate(pop[i]) for i in range(self.population_size)], dtype=float)
            evals_cost += int(self.population_size)

            best_idx = int(np.argmin(costs))
            if float(costs[best_idx]) < best_cost:
                best_cost = float(costs[best_idx])
                best_x = pop[best_idx].copy()
                iter_best_found = gen
                evals_best_found = evals_cost
                time_best_found_sec = elapsed()

            iters = gen
            if iters % self.trace_every == 0:
                checkpoint()

        time_sec = elapsed()
        best_value, best_weight = problem.value_weight(best_x)

        # For GA, final_cost = best of last population
        final_best_idx = int(np.argmin(costs))
        final_cost = float(costs[final_best_idx])
        final_value, final_weight = problem.value_weight(pop[final_best_idx])

        result = RunResult(
            best_solution=best_x,
            best_cost=float(best_cost),
            final_cost=float(final_cost),
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
                "max_iter": self.generations,
                "neighbor_operator": "1pt+bitflip+repair" if self.feasible_only else "1pt+bitflip+penalty",

                "iters": iters,
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
            "current_cost": np.array(trace_curr, dtype=np.float64),
            "best_value": np.array(trace_best_value, dtype=np.float64),
            "best_weight": np.array(trace_best_weight, dtype=np.float64),
        }
        return result, trace
