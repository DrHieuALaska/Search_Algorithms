from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple
import numpy as np


@dataclass
class RunResult:
    best_solution: np.ndarray
    best_cost: float
    final_cost: float
    evals_cost: int
    iters: int
    time_sec: float
    iter_best_found: int
    evals_best_found: int
    time_best_found_sec: float
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseSolver(Protocol):
    """Common interface so all algorithms can be swapped into the same runner."""

    name: str

    def solve(
        self,
        problem: Any,
        init_solution: Optional[np.ndarray] = None,
        logger: Optional[Any] = None,
        **meta: Any
    ) -> Tuple[RunResult, Dict[str, np.ndarray]]:
        ...
