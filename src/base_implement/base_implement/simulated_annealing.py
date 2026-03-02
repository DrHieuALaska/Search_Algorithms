from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar
import random
import math

T = TypeVar("T")  # state type


@dataclass
class SAResult(Generic[T]):
    best_state: T
    best_cost: float
    final_state: T
    final_cost: float
    iters: int
    evals: int


def simulated_annealing(
    *,
    initial_state: T,
    neighbor_fn: Callable[[T, random.Random], T],
    cost_fn: Callable[[T], float],
    T0: float = 1.0,
    Tmin: float = 1e-3,
    alpha: float = 0.99,
    max_iters: int = 50_000,
    rng: Optional[random.Random] = None,
) -> SAResult[T]:
    """Generic Simulated Annealing (minimization)."""
    if rng is None:
        rng = random.Random()

    if T0 <= 0 or Tmin <= 0:
        raise ValueError("T0 and Tmin must be > 0")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0")

    current = initial_state
    current_cost = float(cost_fn(current))
    evals = 1

    best = current
    best_cost = current_cost

    Tcur = float(T0)
    it = 0

    while it < max_iters and Tcur > Tmin:
        cand = neighbor_fn(current, rng)
        cand_cost = float(cost_fn(cand))
        evals += 1

        delta = cand_cost - current_cost
        if delta <= 0:
            accept = True
        else:
            accept = rng.random() < math.exp(-delta / Tcur)

        if accept:
            current = cand
            current_cost = cand_cost
            if current_cost < best_cost:
                best = current
                best_cost = current_cost

        it += 1
        Tcur *= alpha

    return SAResult(
        best_state=best,
        best_cost=best_cost,
        final_state=current,
        final_cost=current_cost,
        iters=it,
        evals=evals,
    )
