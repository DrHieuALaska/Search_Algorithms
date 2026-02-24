from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from core.base import RunResult
from core.logging import RunLogger
from core.rng import make_rng, combine_seeds
from problems.tsp import TSPProblem


class TLBO_TSP:
    """
    Discrete TLBO adaptation for TSP (permutation representation).

    Teacher phase (discrete):
      - For each learner, build a candidate by partially transforming its permutation
        towards the teacher using a swap-sequence (random fraction of swaps).

    Learner phase (discrete):
      - For each learner i, pick partner j. If j is better, i partially transforms
        towards j using the same swap-sequence operator.

    evals_cost counts tour evaluations:
      - init population: pop_size
      - each candidate evaluated: +1
    """

    name = "TLBO"

    def __init__(
        self,
        *,
        pop_size: int = 50,
        iters: int = 500,
        trace_every: int = 10,
    ):
        if pop_size <= 2:
            raise ValueError("pop_size must be > 2")
        if iters <= 0:
            raise ValueError("iters must be > 0")
        if trace_every <= 0:
            raise ValueError("trace_every must be > 0")
        self.pop_size = int(pop_size)
        self.iters = int(iters)
        self.trace_every = int(trace_every)

    @staticmethod
    def _tour_costs(dist: np.ndarray, pop: np.ndarray) -> np.ndarray:
        nxt = np.roll(pop, -1, axis=1)
        return dist[pop, nxt].sum(axis=1)

    @staticmethod
    def _swap_sequence_to_target(src: np.ndarray, target: np.ndarray) -> List[Tuple[int, int]]:
        """Swap list that transforms src into target when applied in order."""
        n = src.shape[0]
        a = src.copy()
        pos = np.empty(n, dtype=np.int32)
        for idx, val in enumerate(a):
            pos[val] = idx

        swaps: List[Tuple[int, int]] = []
        for i in range(n):
            desired = int(target[i])
            if a[i] == desired:
                continue
            j = int(pos[desired])
            ai = int(a[i])
            a[i], a[j] = a[j], a[i]
            pos[ai] = j
            pos[desired] = i
            swaps.append((i, j))
        return swaps

    @staticmethod
    def _apply_prefix_swaps(perm: np.ndarray, swaps: List[Tuple[int, int]], m: int) -> np.ndarray:
        if m <= 0:
            return perm.copy()
        out = perm.copy()
        m = min(m, len(swaps))
        for i in range(m):
            a, b = swaps[i]
            out[a], out[b] = out[b], out[a]
        return out

    def _move_towards(self, src: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        swaps = self._swap_sequence_to_target(src, target)
        if not swaps:
            return src.copy()
        frac = float(rng.random())
        m = max(1, int(frac * len(swaps)))
        return self._apply_prefix_swaps(src, swaps, m)

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

        # Init population deterministic for (instance_seed, seed_algo)
        base = combine_seeds(instance_seed, seed_algo)
        init_rng = make_rng(base)
        pop = np.empty((self.pop_size, problem.n), dtype=np.int32)
        for i in range(self.pop_size):
            pop[i] = problem.random_tour(problem.n, init_rng)

        dist = problem.dist
        costs = self._tour_costs(dist, pop)
        evals_cost = int(self.pop_size)

        best_idx = int(np.argmin(costs))
        best_sol = pop[best_idx].copy()
        best_cost = float(costs[best_idx])

        init_cost = best_cost
        iter_best_found = 0
        evals_best_found = evals_cost
        time_best_found_sec = 0.0

        trace_iter: List[int] = []
        trace_evals: List[int] = []
        trace_time: List[float] = []
        trace_best: List[float] = []

        t_start = time.perf_counter()

        def elapsed() -> float:
            return time.perf_counter() - t_start

        def checkpoint(it: int):
            trace_iter.append(it)
            trace_evals.append(evals_cost)
            trace_time.append(elapsed())
            trace_best.append(best_cost)
            if logger is not None:
                logger.log_trace({
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "iter": it,
                    "evals_cost": evals_cost,
                    "time_sec": trace_time[-1],
                    "temp": "",
                    "best_cost": best_cost,
                    "current_cost": best_cost,
                })

        checkpoint(0)

        for it in range(1, self.iters + 1):
            # Teacher phase
            teacher_idx = int(np.argmin(costs))
            teacher = pop[teacher_idx]

            for i in range(self.pop_size):
                if i == teacher_idx:
                    continue
                cand = self._move_towards(pop[i], teacher, rng)
                cand_cost = float(dist[cand, np.roll(cand, -1)].sum())
                evals_cost += 1
                if cand_cost < float(costs[i]):
                    pop[i] = cand
                    costs[i] = cand_cost
                    if cand_cost < best_cost:
                        best_cost = cand_cost
                        best_sol = cand.copy()
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
                    cand_cost = float(dist[cand, np.roll(cand, -1)].sum())
                    evals_cost += 1
                    if cand_cost < float(costs[i]):
                        pop[i] = cand
                        costs[i] = cand_cost
                        if cand_cost < best_cost:
                            best_cost = cand_cost
                            best_sol = cand.copy()
                            iter_best_found = it
                            evals_best_found = evals_cost
                            time_best_found_sec = elapsed()

            if it % self.trace_every == 0:
                checkpoint(it)

        time_sec = elapsed()

        result = RunResult(
            best_solution=best_sol,
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
                "neighbor_operator": "swap-seq",

                "iters": self.iters,
                "evals_cost": evals_cost,
                "time_sec": time_sec,

                "init_cost": init_cost,
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
        }
        return result, trace
