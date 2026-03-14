from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar
import random
import math

N = TypeVar("N")  # node type (hashable)


@dataclass
class PathResult(Generic[N]):
    found: bool
    path: List[N]
    visited: int


def bfs(
    *,
    start: N,
    goal_test: Callable[[N], bool],
    neighbors_fn: Callable[[N], Iterable[N]],
) -> PathResult[N]:
    """Breadth-First Search on an implicit unweighted graph."""
    from collections import deque

    if goal_test(start):
        return PathResult(True, [start], visited=1)

    q = deque([start])
    parent: Dict[N, Optional[N]] = {start: None}
    visited = 0

    while q:
        u = q.popleft()
        visited += 1
        for v in neighbors_fn(u):
            if v in parent:
                continue
            parent[v] = u
            if goal_test(v):
                path: List[N] = [v]
                cur = u
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return PathResult(True, path, visited=visited)
            q.append(v)

    return PathResult(False, [], visited=visited)
