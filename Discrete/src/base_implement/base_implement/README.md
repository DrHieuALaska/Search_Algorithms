# Base Implementations (Templates)

This folder contains clean, minimal **base implementations** of common optimization and search algorithms.

These are **generic templates**:
- Optimization algorithms (SA, HC, TLBO) operate on an abstract *state* and require you to provide:
  - `neighbor_fn(state, rng)` or `neighbors_fn(state)` (depending on algorithm)
  - `cost_fn(state)` (minimization by default)
- Graph search algorithms (BFS, DFS, A*) operate on an implicit graph and require:
  - `neighbors_fn(node)` -> iterable of neighbor nodes
  - `goal_test(node)` -> bool
  - (A*) `heuristic(node)` -> nonnegative estimate to goal
  - optionally `edge_cost(u, v)` -> float (defaults to 1)

Files:
- `simulated_annealing.py`
- `hill_climbing.py`
- `tlbo.py`
- `bfs.py`
- `dfs.py`
- `astar.py`

All implementations are standard-library-only (no external dependencies).
