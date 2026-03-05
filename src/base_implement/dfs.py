from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar
import random
import math

N = TypeVar("N")


@dataclass
class PathResult(Generic[N]):
    found: bool
    path: List[N]
    visited: int


def dfs(
    *,
    start: N,
    goal_test: Callable[[N], bool],
    neighbors_fn: Callable[[N], Iterable[N]],
    depth_limit: Optional[int] = None,
) -> PathResult[N]:
    """Depth-First Search (optionally depth-limited) on an implicit graph."""
    stack: List[Tuple[N, int]] = [(start, 0)]
    parent: Dict[N, Optional[N]] = {start: None}
    visited = 0

    while stack:
        u, depth = stack.pop()
        visited += 1
        if goal_test(u):
            path: List[N] = [u]
            cur = parent[u]
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return PathResult(True, path, visited=visited)

        if depth_limit is not None and depth >= depth_limit:
            continue

        for v in neighbors_fn(u):
            if v in parent:
                continue
            parent[v] = u
            stack.append((v, depth + 1))

    return PathResult(False, [], visited=visited)
