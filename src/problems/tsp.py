from __future__ import annotations
import numpy as np


class TSPProblem:
    """Euclidean TSP instance with fast evaluation helpers."""

    def __init__(
        self,
        n_cities: int,
        *,
        instance_seed: int = 0,
        scale: float = 100.0,
        cities: np.ndarray | None = None,
        distance_matrix: np.ndarray | None = None,
    ):
        if distance_matrix is not None:
            dist = np.asarray(distance_matrix, dtype=float)
            if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
                raise ValueError("distance_matrix must be square (n,n)")
            self.n = int(dist.shape[0])
            self.dist = dist
            self.cities = None if cities is None else np.asarray(cities, dtype=float)
            return

        if cities is None:
            rng = np.random.default_rng(int(instance_seed))
            cities = rng.random((int(n_cities), 2)) * float(scale)
        else:
            cities = np.asarray(cities, dtype=float)
            if cities.ndim != 2 or cities.shape[1] != 2:
                raise ValueError("cities must have shape (n,2)")
            n_cities = cities.shape[0]

        self.n = int(n_cities)
        self.cities = cities
        self.dist = self._compute_distance_matrix(cities)

    @staticmethod
    def _compute_distance_matrix(cities: np.ndarray) -> np.ndarray:
        # Vectorized Euclidean distances
        diff = cities[:, None, :] - cities[None, :, :]
        return np.sqrt((diff * diff).sum(axis=2))

    def evaluate(self, tour: np.ndarray) -> float:
        """Full tour length (cycle)."""
        t = np.asarray(tour, dtype=np.int32)
        return float(self.dist[t, np.roll(t, -1)].sum())

    def delta_2opt(self, tour: np.ndarray, i: int, k: int) -> float:
        """
        O(1) delta for 2-opt reversal on indices [i..k] inclusive (symmetric distance).
        Only two boundary edges change in the symmetric case.
        """
        n = self.n
        t = tour
        a = t[i - 1] if i > 0 else t[n - 1]
        b = t[i]
        c = t[k]
        d = t[k + 1] if (k + 1) < n else t[0]

        old = self.dist[a, b] + self.dist[c, d]
        new = self.dist[a, c] + self.dist[b, d]
        return float(new - old)

    @staticmethod
    def random_tour(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.permutation(int(n)).astype(np.int32)
