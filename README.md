# TSP Optimization Project Scaffold (Option A)

Clean modular structure for TSP + Simulated Annealing (SA), designed to scale to more algorithms (TLBO/ACO/GA/...).

## Structure
- `src/problems/tsp.py` : TSPProblem (instance generation, evaluation, 2-opt delta)
- `src/solvers/sa_tsp.py` : SimulatedAnnealingTSP solver (fast O(1) 2-opt delta)
- `src/core/base.py` : BaseSolver interface + RunResult dataclass
- `src/core/logging.py` : RunLogger (run-level + trace-level CSV)
- `src/core/rng.py` : seed utilities (instance_seed vs seed_algo)
- `src/experiments/run_tsp.py` : CLI runner to generate runs and log CSV

## Quick start
From the project folder:

```bash
python src/experiments/run_tsp.py --n_list 5 15 30 50 --instances 10 --runs_per_instance 10   --run_csv runs.csv --trace_csv trace.csv
```

This will generate:
- `runs.csv` (one row per run) for statistics (Wilcoxon, CI, mean/std, median/IQR)
- `trace.csv` (checkpoint rows) for convergence plots (best_cost vs evals/time)

## Pairing setup (important for statistics)
- `instance_seed` controls city generation => defines the TSP instance.
- `seed_algo` controls solver randomness + init tour => defines the run replicate.

For fair paired tests, compare algorithms on the same `(n, instance_seed, seed_algo)`.

## Notes
- Distances are Euclidean (symmetric). The 2-opt delta uses the symmetric shortcut (only 2 edges change).
- `evals_cost` counts objective evaluations:
  - SA counts 1 initial evaluation + 1 per attempted move (even though delta is O(1)).


## Algorithms
- SA: `src/solvers/sa_tsp.py`
- TLBO (discrete swap-sequence): `src/solvers/tlbo_tsp.py`
