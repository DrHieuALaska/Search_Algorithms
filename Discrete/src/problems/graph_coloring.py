from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


class GraphColoringProblem:
    """k-Coloring as conflict minimization.

    A solution is an int array `colors` of shape (n_nodes,), values in [0, k-1].
    Objective (minimize): number of conflicting edges where endpoints share the same color.

    For small graphs, DFS solver may find a 0-conflict coloring (proper k-coloring).
    """

    def __init__(self, adj: List[List[int]], k: int, instance_seed: int = 0):
        if k <= 0:
            raise ValueError("k must be > 0")
        self.adj = adj
        self.n = len(adj)
        self.k = int(k)
        self.instance_seed = int(instance_seed)

        # Precompute undirected edge list (i<j) for evaluation
        edges: List[Tuple[int,int]] = []
        for i in range(self.n):
            for j in self.adj[i]:
                if i < j:
                    edges.append((i, j))
        self.edges = edges

        # Degrees (useful for heuristics/ordering)
        self.deg = np.array([len(a) for a in adj], dtype=np.int32)

    @staticmethod
    def random_instance(
        n_nodes: int,
        k: int,
        edge_prob: float,
        instance_seed: int = 0,
        ensure_connected: bool = False,
    ) -> "GraphColoringProblem":
        """Generate an Erdos–Renyi undirected graph G(n,p)."""
        if n_nodes <= 0:
            raise ValueError("n_nodes must be > 0")
        if not (0.0 <= edge_prob <= 1.0):
            raise ValueError("edge_prob must be in [0,1]")

        rng = np.random.default_rng(int(instance_seed))
        adj = [[] for _ in range(int(n_nodes))]
        for i in range(int(n_nodes)):
            for j in range(i + 1, int(n_nodes)):
                if rng.random() < float(edge_prob):
                    adj[i].append(j)
                    adj[j].append(i)

        if ensure_connected and n_nodes > 1:
            # simple connect components by adding edges between consecutive nodes in a random permutation
            perm = rng.permutation(int(n_nodes))
            for a, b in zip(perm[:-1], perm[1:]):
                a = int(a); b = int(b)
                if b not in adj[a]:
                    adj[a].append(b)
                    adj[b].append(a)

        return GraphColoringProblem(adj=adj, k=k, instance_seed=instance_seed)

    def evaluate(self, colors: np.ndarray) -> int:
        """Return number of conflicting edges (i<j) where colors[i]==colors[j]."""
        c = np.asarray(colors, dtype=np.int32)
        conflicts = 0
        for i, j in self.edges:
            if c[i] == c[j]:
                conflicts += 1
        return int(conflicts)

    def delta_recolor(self, colors: np.ndarray, v: int, new_color: int) -> int:
        """O(deg(v)) delta conflicts for changing vertex v to new_color."""
        old = int(colors[v])
        if old == new_color:
            return 0
        delta = 0
        for u in self.adj[v]:
            cu = int(colors[u])
            if cu == old:
                delta -= 1
            if cu == new_color:
                delta += 1
        return int(delta)

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, self.k, size=self.n, dtype=np.int32)

    def greedy_init(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Greedy coloring (may use >k colors internally, then mod k). For init only."""
        if rng is None:
            rng = np.random.default_rng(self.instance_seed)

        order = np.argsort(-self.deg)  # high degree first
        colors = -np.ones(self.n, dtype=np.int32)
        for v in order:
            used = set(colors[u] for u in self.adj[int(v)] if colors[u] >= 0)
            for col in range(self.k):
                if col not in used:
                    colors[int(v)] = col
                    break
            if colors[int(v)] < 0:
                colors[int(v)] = int(rng.integers(0, self.k))
        return colors
