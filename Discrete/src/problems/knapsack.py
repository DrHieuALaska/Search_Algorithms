from __future__ import annotations
import numpy as np


class KnapsackProblem:
    """
    0/1 Knapsack (maximize total value subject to total weight <= capacity).

    We use a *minimization* cost for compatibility with the codebase:
        cost(x) = -value(x) + penalty_lambda * max(0, weight(x) - capacity)

    If you enable repair, solutions are forced feasible and penalty becomes 0.
    """

    def __init__(
        self,
        *,
        weights: np.ndarray,
        values: np.ndarray,
        capacity: float,
        penalty_lambda: float | None = None,
    ):
        w = np.asarray(weights, dtype=float)
        v = np.asarray(values, dtype=float)
        if w.ndim != 1 or v.ndim != 1 or w.shape[0] != v.shape[0]:
            raise ValueError("weights and values must be 1D arrays with same length")
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if np.any(w <= 0) or np.any(v < 0):
            raise ValueError("weights must be > 0 and values must be >= 0")

        self.n = int(w.shape[0])
        self.weights = w
        self.values = v
        self.capacity = float(capacity)

        if penalty_lambda is None:
            # A reasonable default scale: large enough to discourage infeasible solutions.
            self.penalty_lambda = float(10.0 * np.max(v) / (np.min(w) + 1e-12))
        else:
            self.penalty_lambda = float(penalty_lambda)

    @staticmethod
    def random_instance(
        n_items: int,
        *,
        instance_seed: int = 0,
        weight_range: tuple[int, int] = (1, 50),
        value_range: tuple[int, int] = (1, 100),
        capacity_ratio: float = 0.4,
    ) -> "KnapsackProblem":
        """Generate a random knapsack instance."""
        rng = np.random.default_rng(int(instance_seed))
        w = rng.integers(weight_range[0], weight_range[1] + 1, size=n_items).astype(float)
        v = rng.integers(value_range[0], value_range[1] + 1, size=n_items).astype(float)
        cap = float(capacity_ratio * w.sum())
        return KnapsackProblem(weights=w, values=v, capacity=cap, penalty_lambda=None)

    def evaluate(self, x: np.ndarray) -> float:
        """Return cost (minimize)."""
        xb = np.asarray(x, dtype=np.int8)
        if xb.ndim != 1 or xb.shape[0] != self.n:
            raise ValueError("x must have shape (n_items,)")
        total_w = float((xb * self.weights).sum())
        total_v = float((xb * self.values).sum())
        violation = max(0.0, total_w - self.capacity)
        return -total_v + self.penalty_lambda * violation

    def value_weight(self, x: np.ndarray) -> tuple[float, float]:
        xb = np.asarray(x, dtype=np.int8)
        total_w = float((xb * self.weights).sum())
        total_v = float((xb * self.values).sum())
        return total_v, total_w

    def repair(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Make solution feasible by removing items with worst value/weight first."""
        xb = np.asarray(x, dtype=np.int8).copy()
        if rng is None:
            rng = np.random.default_rng(0)

        # while overweight, drop items with lowest ratio; break ties randomly
        total_w = float((xb * self.weights).sum())
        if total_w <= self.capacity:
            return xb

        ratio = self.values / self.weights
        # indices currently included
        while total_w > self.capacity:
            ones = np.flatnonzero(xb)
            if ones.size == 0:
                break
            r = ratio[ones]
            # pick among the worst few to add randomness
            worst = ones[np.argsort(r)]
            cut = max(1, min(5, worst.size))
            drop = int(worst[rng.integers(0, cut)])
            xb[drop] = 0
            total_w -= float(self.weights[drop])

        return xb

    def random_solution(self, rng: np.random.Generator, *, feasible: bool = True) -> np.ndarray:
        x = rng.integers(0, 2, size=self.n, dtype=np.int8)
        return self.repair(x, rng) if feasible else x
