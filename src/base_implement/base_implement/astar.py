from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar
import random
import math

N = TypeVar("N")


@dataclass
class AStarResult(Generic[N]):
    found: bool
    path: List[N]
    cost: float
    expanded: int


def astar(
    *,
    start: N,
    goal_test: Callable[[N], bool],
    neighbors_fn: Callable[[N], Iterable[N]],
    heuristic: Callable[[N], float],
    edge_cost: Optional[Callable[[N, N], float]] = None,
) -> AStarResult[N]:
    """A* search on an implicit graph."""
    import heapq

    if edge_cost is None:
        def edge_cost(u: N, v: N) -> float:  # type: ignore
            return 1.0

    g: Dict[N, float] = {start: 0.0}
    parent: Dict[N, Optional[N]] = {start: None}

    pq: List[Tuple[float, int, N]] = []
    tie = 0
    heapq.heappush(pq, (float(heuristic(start)), tie, start))

    expanded = 0
    closed: set[N] = set()

    while pq:
        _, _, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1

        if goal_test(u):
            path: List[N] = [u]
            cur = parent[u]
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return AStarResult(True, path, g[u], expanded)

        for v in neighbors_fn(u):
            if v in closed:
                continue
            tentative = g[u] + float(edge_cost(u, v))
            if v not in g or tentative < g[v]:
                g[v] = tentative
                parent[v] = u
                tie += 1
                heapq.heappush(pq, (tentative + float(heuristic(v)), tie, v))

    return AStarResult(False, [], float("inf"), expanded)
