from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar
import random
import math

T = TypeVar("T")


@dataclass
class TLBOResult(Generic[T]):
    best_state: T
    best_cost: float
    iters: int
    evals: int


def tlbo(
    *,
    init_population_fn: Callable[[int, random.Random], List[T]],
    cost_fn: Callable[[T], float],
    move_towards_fn: Callable[[T, T, random.Random], T],
    pop_size: int = 50,
    iters: int = 500,
    rng: Optional[random.Random] = None,
) -> TLBOResult[T]:
    """Generic TLBO template (minimization)."""
    if rng is None:
        rng = random.Random()
    if pop_size <= 2:
        raise ValueError("pop_size must be > 2")
    if iters <= 0:
        raise ValueError("iters must be > 0")

    pop = init_population_fn(pop_size, rng)
    if len(pop) != pop_size:
        raise ValueError("init_population_fn returned wrong population size")

    costs = [float(cost_fn(x)) for x in pop]
    evals = pop_size

    best_idx = min(range(pop_size), key=lambda i: costs[i])
    best = pop[best_idx]
    best_cost = float(costs[best_idx])

    for _ in range(1, iters + 1):
        teacher_idx = min(range(pop_size), key=lambda i: costs[i])
        teacher = pop[teacher_idx]

        for i in range(pop_size):
            if i == teacher_idx:
                continue
            cand = move_towards_fn(pop[i], teacher, rng)
            c = float(cost_fn(cand))
            evals += 1
            if c < costs[i]:
                pop[i] = cand
                costs[i] = c
                if c < best_cost:
                    best_cost = c
                    best = cand

        for i in range(pop_size):
            j = rng.randrange(pop_size - 1)
            if j >= i:
                j += 1
            if costs[j] < costs[i]:
                cand = move_towards_fn(pop[i], pop[j], rng)
                c = float(cost_fn(cand))
                evals += 1
                if c < costs[i]:
                    pop[i] = cand
                    costs[i] = c
                    if c < best_cost:
                        best_cost = c
                        best = cand

    return TLBOResult(best_state=best, best_cost=best_cost, iters=iters, evals=evals)
