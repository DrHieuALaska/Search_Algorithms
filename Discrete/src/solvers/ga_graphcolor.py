from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from core.base import RunResult
from core.logging import RunLogger
from core.rng import make_rng, combine_seeds
from problems.graph_coloring import GraphColoringProblem


class GeneticAlgorithmGraphColoring:
    """GA for k-coloring (minimize conflict edges)."""

    name = "GA_GC"

    def __init__(
        self,
        *,
        population_size: int = 100,
        generations: int = 500,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.02,
        tournament_k: int = 10,
        trace_every: int = 10,
    ):
        if population_size <= 2:
            raise ValueError("population_size must be > 2")
        if generations <= 0:
            raise ValueError("generations must be > 0")
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0,1]")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0,1]")
        if tournament_k <= 0 or tournament_k > population_size:
            raise ValueError("tournament_k must be in [1,population_size]")
        if trace_every <= 0:
            raise ValueError("trace_every must be > 0")

        self.population_size = int(population_size)
        self.generations = int(generations)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.tournament_k = int(tournament_k)
        self.trace_every = int(trace_every)

    @staticmethod
    def _fitness(problem: GraphColoringProblem, pop: np.ndarray) -> np.ndarray:
        # vectorization: compute conflicts by counting same-color edges
        # For simplicity, do per-individual loop (graphs are moderate)
        fit = np.empty(pop.shape[0], dtype=np.int32)
        for i in range(pop.shape[0]):
            fit[i] = problem.evaluate(pop[i])
        return fit

    def _tournament(self, pop: np.ndarray, fit: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(pop.shape[0], self.tournament_k, replace=False)
        best = idx[int(np.argmin(fit[idx]))]
        return pop[best]

    def _crossover(self, a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = a.shape[0]
        if rng.random() >= self.crossover_rate:
            return a.copy()
        cut = int(rng.integers(1, n))
        child = np.empty(n, dtype=np.int32)
        child[:cut] = a[:cut]
        child[cut:] = b[cut:]
        return child

    def _mutate(self, x: np.ndarray, k: int, rng: np.random.Generator) -> None:
        # per-gene recolor with probability mutation_rate
        if self.mutation_rate <= 0:
            return
        mask = rng.random(x.shape[0]) < self.mutation_rate
        if not mask.any():
            return
        x[mask] = rng.integers(0, k, size=int(mask.sum()), dtype=np.int32)

    def solve(
        self,
        problem: GraphColoringProblem,
        init_solution: Optional[np.ndarray] = None,
        logger: Optional[RunLogger] = None,
        **meta: Any
    ) -> Tuple[RunResult, Dict[str, np.ndarray]]:
        experiment_id = str(meta.get("experiment_id", "exp_gc_001"))
        run_id = str(meta.get("run_id", "run_001"))
        instance_seed = int(meta.get("instance_seed", 0))
        seed_algo = int(meta.get("seed_algo", 0))
        code_version = str(meta.get("code_version", ""))

        rng = make_rng(seed_algo)

        init_rng = make_rng(combine_seeds(instance_seed, seed_algo))
        pop = init_rng.integers(0, problem.k, size=(self.population_size, problem.n), dtype=np.int32)
        if init_solution is not None:
            pop[0] = np.asarray(init_solution, dtype=np.int32)

        fit = self._fitness(problem, pop)
        evals_cost = int(self.population_size)

        best_idx = int(np.argmin(fit))
        best_sol = pop[best_idx].copy()
        best_cost = int(fit[best_idx])
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
        def elapsed(): return time.perf_counter() - t_start

        def checkpoint():
            trace_iter.append(iters)
            trace_evals.append(evals_cost)
            trace_time.append(elapsed())
            trace_best.append(best_cost)
            trace_curr.append(int(np.min(fit)))
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
                })

        checkpoint()

        for g in range(self.generations):
            offspring = np.empty_like(pop)
            for i in range(self.population_size):
                p1 = self._tournament(pop, fit, rng)
                p2 = self._tournament(pop, fit, rng)
                child = self._crossover(p1, p2, rng)
                self._mutate(child, problem.k, rng)
                offspring[i] = child

            fit_off = self._fitness(problem, offspring)
            evals_cost += int(self.population_size)

            combined = np.vstack([pop, offspring])
            combined_fit = np.concatenate([fit, fit_off])
            keep = np.argsort(combined_fit)[: self.population_size]
            pop = combined[keep]
            fit = combined_fit[keep]

            if int(fit[0]) < best_cost:
                best_cost = int(fit[0])
                best_sol = pop[0].copy()
                iter_best_found = g + 1
                evals_best_found = evals_cost
                time_best_found_sec = elapsed()

            iters = g + 1
            if iters % self.trace_every == 0:
                checkpoint()
            if best_cost == 0:
                break

        checkpoint()
        time_sec = elapsed()

        result = RunResult(
            best_solution=best_sol,
            best_cost=float(best_cost),
            final_cost=float(int(fit[0])),
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
                "experiment_id": experiment_id,
                "run_id": run_id,
                "timestamp_utc": ts_utc,
                "algorithm": self.name,
                "problem": "GRAPH_COLORING",
                "n_cities": problem.n,
                "instance_seed": instance_seed,
                "coord_scale": "",
                "distance_type": "graph",
                "init_temp_T0": "",
                "min_temp_Tmin": "",
                "alpha": "",
                "steps_per_temp": "",
                "max_iter": self.generations,
                "neighbor_operator": f"1pt-xover+recolor(mut={self.mutation_rate})",
                "iters": iters,
                "evals_cost": evals_cost,
                "time_sec": time_sec,
                "init_cost": init_cost,
                "final_cost": int(fit[0]),
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
