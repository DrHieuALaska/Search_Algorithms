from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar
import random
import math

T = TypeVar("T")


@dataclass
class HCResult(Generic[T]):
    best_state: T
    best_cost: float
    final_state: T
    final_cost: float
    iters: int
    evals: int


def hill_climbing_first_improvement(
    *,
    initial_state: T,
    neighbor_fn: Callable[[T, random.Random], T],
    cost_fn: Callable[[T], float],
    max_iters: int = 50_000,
    rng: Optional[random.Random] = None,
) -> HCResult[T]:
    """Hill Climbing (first-improvement) for minimization."""
    if rng is None:
        rng = random.Random()
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0")

    current = initial_state
    current_cost = float(cost_fn(current))
    evals = 1

    best = current
    best_cost = current_cost

    it = 0
    while it < max_iters:
        cand = neighbor_fn(current, rng)
        cand_cost = float(cost_fn(cand))
        evals += 1
        it += 1

        if cand_cost < current_cost:
            current = cand
            current_cost = cand_cost
            if current_cost < best_cost:
                best = current
                best_cost = current_cost

    return HCResult(best, best_cost, current, current_cost, it, evals)


def hill_climbing_best_improvement(
    *,
    initial_state: T,
    neighbors_fn: Callable[[T], Iterable[T]],
    cost_fn: Callable[[T], float],
    max_iters: int = 10_000,
) -> HCResult[T]:
    """Hill Climbing (steepest-descent) for minimization."""
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0")

    current = initial_state
    current_cost = float(cost_fn(current))
    evals = 1

    best = current
    best_cost = current_cost

    it = 0
    while it < max_iters:
        it += 1
        best_neighbor = None
        best_neighbor_cost = current_cost

        for nb in neighbors_fn(current):
            c = float(cost_fn(nb))
            evals += 1
            if c < best_neighbor_cost:
                best_neighbor_cost = c
                best_neighbor = nb

        if best_neighbor is None:
            break

        current = best_neighbor
        current_cost = best_neighbor_cost
        if current_cost < best_cost:
            best = current
            best_cost = current_cost

    return HCResult(best, best_cost, current, current_cost, it, evals)
