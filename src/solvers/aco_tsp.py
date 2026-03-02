from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from core.base import RunResult
from core.logging import RunLogger
from core.rng import make_rng
from problems.tsp import TSPProblem


class ACO_TSP:
    """Ant Colony Optimization (Ant System) for TSP (symmetric Euclidean)."""

    name = "ACO"

    def __init__(
        self,
        *,
        n_ants: int = 30,
        iters: int = 1667,
        alpha: float = 1.0,
        beta: float = 3.0,
        rho: float = 0.1,     # evaporation rate
        Q: float = 100.0,     # pheromone deposit constant
        trace_every: int = 10,
        deposit_best_only: bool = False,
    ):
        if n_ants <= 0:
            raise ValueError("n_ants must be > 0")
        if iters <= 0:
            raise ValueError("iters must be > 0")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if beta < 0:
            raise ValueError("beta must be >= 0")
        if not (0 < rho < 1):
            raise ValueError("rho must be in (0,1)")
        if Q <= 0:
            raise ValueError("Q must be > 0")
        if trace_every <= 0:
            raise ValueError("trace_every must be > 0")

        self.n_ants = int(n_ants)
        self.iters = int(iters)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.Q = float(Q)
        self.trace_every = int(trace_every)
        self.deposit_best_only = bool(deposit_best_only)

    def _construct_tour(self, dist: np.ndarray, tau: np.ndarray, eta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = dist.shape[0]
        tour = np.empty(n, dtype=np.int32)
        start = int(rng.integers(0, n))
        tour[0] = start
        visited = np.zeros(n, dtype=bool)
        visited[start] = True

        for t in range(1, n):
            cur = tour[t - 1]
            unvis = np.flatnonzero(~visited)
            # desirability
            desir = (tau[cur, unvis] ** self.alpha) * (eta[cur, unvis] ** self.beta)
            s = desir.sum()
            if s <= 0 or not np.isfinite(s):
                nxt = int(rng.choice(unvis))
            else:
                p = desir / s
                nxt = int(rng.choice(unvis, p=p))
            tour[t] = nxt
            visited[nxt] = True
        return tour

    @staticmethod
    def _tour_cost(dist: np.ndarray, tour: np.ndarray) -> float:
        return float(dist[tour, np.roll(tour, -1)].sum())

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

        dist = problem.dist
        n = problem.n

        # heuristic info
        eps = 1e-12
        eta = 1.0 / (dist + eps)
        np.fill_diagonal(eta, 0.0)

        # pheromone init: small constant
        tau = np.full((n, n), 1.0, dtype=np.float64)
        np.fill_diagonal(tau, 0.0)

        best_tour = None
        best_cost = float("inf")

        evals_cost = 0
        it = 0

        iter_best_found = 0
        evals_best_found = 0
        time_best_found_sec = 0.0

        trace_iter: List[int] = []
        trace_evals: List[int] = []
        trace_time: List[float] = []
        trace_best: List[float] = []
        trace_curr: List[float] = []

        t_start = time.perf_counter()

        def elapsed() -> float:
            return time.perf_counter() - t_start

        def checkpoint(curr_best: float):
            trace_iter.append(it)
            trace_evals.append(evals_cost)
            trace_time.append(elapsed())
            trace_best.append(best_cost)
            trace_curr.append(curr_best)

            if logger is not None:
                logger.log_trace({
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "iter": it,
                    "evals_cost": evals_cost,
                    "time_sec": trace_time[-1],
                    "temp": "",  # not applicable
                    "best_cost": best_cost,
                    "current_cost": curr_best,
                })

        # initial checkpoint
        checkpoint(curr_best=float("inf"))

        for it_ in range(1, self.iters + 1):
            it = it_
            tours = np.empty((self.n_ants, n), dtype=np.int32)
            costs = np.empty(self.n_ants, dtype=np.float64)

            for a in range(self.n_ants):
                tour = self._construct_tour(dist, tau, eta, rng)
                c = self._tour_cost(dist, tour)
                tours[a] = tour
                costs[a] = c
            evals_cost += self.n_ants

            # best of iteration
            idx = int(np.argmin(costs))
            iter_best_cost = float(costs[idx])

            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_tour = tours[idx].copy()
                iter_best_found = it
                evals_best_found = evals_cost
                time_best_found_sec = elapsed()

            # pheromone evaporation
            tau *= (1.0 - self.rho)

            # deposit
            if self.deposit_best_only:
                dep_indices = [idx]
            else:
                dep_indices = list(range(self.n_ants))

            for a in dep_indices:
                tour = tours[a]
                c = float(costs[a])
                if c <= 0:
                    continue
                delta = self.Q / c
                edges_from = tour
                edges_to = np.roll(tour, -1)
                tau[edges_from, edges_to] += delta
                tau[edges_to, edges_from] += delta  # symmetric

            np.fill_diagonal(tau, 0.0)

            if it % self.trace_every == 0:
                checkpoint(curr_best=iter_best_cost)

        if best_tour is None:
            # fallback: random tour
            best_tour = problem.random_tour(n, rng)
            best_cost = problem.evaluate(best_tour)
            evals_cost += 1

        time_sec = elapsed()
        result = RunResult(
            best_solution=best_tour,
            best_cost=float(best_cost),
            final_cost=float(best_cost),
            evals_cost=int(evals_cost),
            iters=int(self.iters),
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
                "max_iter": self.iters,
                "neighbor_operator": f"ACO(ants={self.n_ants},beta={self.beta},rho={self.rho},Q={self.Q})",

                "iters": self.iters,
                "evals_cost": evals_cost,
                "time_sec": time_sec,

                "init_cost": "",
                "final_cost": best_cost,
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
