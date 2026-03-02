from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from core.base import RunResult
from core.logging import RunLogger
from core.rng import make_rng, combine_seeds
from problems.tsp import TSPProblem


class GeneticAlgorithmTSP:
    """Genetic Algorithm for TSP using Order Crossover (OX) + swap mutation."""

    name = "GA"

    def __init__(
        self,
        *,
        population_size: int = 100,
        generations: int = 400,
        mutation_rate: float = 0.2,
        tournament_k: int = 10,
        trace_every: int = 10,
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

    # -------------------------------
    # Operators
    # -------------------------------

    def _mutate_swap(self, route: np.ndarray, rng: np.random.Generator) -> None:
        if rng.random() < self.mutation_rate:
            i, j = rng.choice(route.shape[0], 2, replace=False)
            route[i], route[j] = route[j], route[i]

    def _order_crossover(self, mom: np.ndarray, dad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Order crossover (OX) for permutations."""
        n = mom.shape[0]
        a, b = sorted(rng.choice(n, 2, replace=False))
        child = np.full(n, -1, dtype=np.int32)
        child[a:b] = mom[a:b]

        used = np.zeros(n, dtype=bool)
        used[child[a:b]] = True

        fill = []
        for x in dad:
            xi = int(x)
            if not used[xi]:
                fill.append(xi)
                used[xi] = True

        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[idx]
                idx += 1
        return child

    def _tournament_selection(
        self,
        population: np.ndarray,      # shape (P, n)
        fitness: np.ndarray,         # shape (P,)
        rng: np.random.Generator
    ) -> np.ndarray:
        idx = rng.choice(population.shape[0], self.tournament_k, replace=False)
        best = idx[int(np.argmin(fitness[idx]))]
        return population[best]

    # -------------------------------
    # Solve
    # -------------------------------

    def solve(
        self,
        problem: TSPProblem,
        init_solution: Optional[np.ndarray] = None,
        logger: Optional[RunLogger] = None,
        **meta: Any
    ) -> Tuple[RunResult, Dict[str, np.ndarray]]:
        experiment_id = str(meta.get("experiment_id", "exp_001"))
        run_id = str(meta.get("run_id", "run_001"))
        instance_seed = int(meta.get("instance_seed", 0))
        seed_algo = int(meta.get("seed_algo", 0))
        coord_scale = float(meta.get("coord_scale", 100.0))
        code_version = str(meta.get("code_version", ""))

        rng = make_rng(seed_algo)

        # Init population should be reproducible per (instance_seed, seed_algo) for paired comparisons
        init_rng = make_rng(combine_seeds(instance_seed, seed_algo))
        pop = np.empty((self.population_size, problem.n), dtype=np.int32)
        for i in range(self.population_size):
            pop[i] = problem.random_tour(problem.n, init_rng)

        if init_solution is not None:
            pop[0] = np.asarray(init_solution, dtype=np.int32)

        # Fitness (vectorized)
        dist = problem.dist
        nxt = np.roll(pop, -1, axis=1)
        fitness = dist[pop, nxt].sum(axis=1).astype(float)
        evals_cost = int(self.population_size)

        best_idx = int(np.argmin(fitness))
        best_cost = float(fitness[best_idx])
        best_sol = pop[best_idx].copy()
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

        t_start = time.perf_counter()

        def elapsed() -> float:
            return time.perf_counter() - t_start

        def checkpoint():
            trace_iter.append(iters)
            trace_evals.append(evals_cost)
            trace_time.append(elapsed())
            trace_best.append(best_cost)
            trace_curr.append(float(np.min(fitness)))

            if logger is not None:
                logger.log_trace({
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "iter": iters,
                    "evals_cost": evals_cost,
                    "time_sec": trace_time[-1],
                    "temp": "",  # not applicable
                    "best_cost": best_cost,
                    "current_cost": trace_curr[-1],
                })

        checkpoint()

        # Evolution loop
        for gen in range(self.generations):
            # create offspring
            offspring = np.empty_like(pop)
            for i in range(self.population_size):
                mom = self._tournament_selection(pop, fitness, rng)
                dad = self._tournament_selection(pop, fitness, rng)
                child = self._order_crossover(mom, dad, rng)
                self._mutate_swap(child, rng)
                offspring[i] = child

            # evaluate offspring
            nxt_off = np.roll(offspring, -1, axis=1)
            fit_off = dist[offspring, nxt_off].sum(axis=1).astype(float)
            evals_cost += int(self.population_size)

            # elitist selection: keep best P from (pop + offspring)
            combined = np.vstack([pop, offspring])
            combined_fit = np.concatenate([fitness, fit_off])
            keep = np.argsort(combined_fit)[: self.population_size]
            pop = combined[keep]
            fitness = combined_fit[keep]

            # update best
            if float(fitness[0]) < best_cost:
                best_cost = float(fitness[0])
                best_sol = pop[0].copy()
                iter_best_found = gen + 1
                evals_best_found = evals_cost
                time_best_found_sec = elapsed()

            iters = gen + 1
            if iters % self.trace_every == 0:
                checkpoint()

        time_sec = elapsed()

        result = RunResult(
            best_solution=best_sol,
            best_cost=float(best_cost),
            final_cost=float(fitness[0]),
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

                # SA-only params: blank
                "init_temp_T0": "",
                "min_temp_Tmin": "",
                "alpha": "",
                "steps_per_temp": "",
                "max_iter": self.generations,
                "neighbor_operator": "OX+swap",

                "iters": iters,
                "evals_cost": evals_cost,
                "time_sec": time_sec,

                "init_cost": init_cost,
                "final_cost": float(fitness[0]),
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
