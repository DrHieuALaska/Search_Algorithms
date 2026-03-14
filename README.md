# Search & Optimization Algorithm Benchmark

A systematic empirical study of metaheuristic and exact search algorithms on three combinatorial optimization problems, with full parameter sensitivity analysis, convergence visualization, and normalized evaluation budgets.

---

## Problems

| Problem | Type | Objective | Instance size |
|---|---|---|---|
| **TSP** (Traveling Salesman) | Minimization | Total Euclidean tour length | n = 30 cities |
| **Knapsack** (0/1) | Maximization | Total value subject to capacity | n = 100 items |
| **Graph Coloring** | Minimization | Conflicting edges in k-coloring | n = 30 nodes, k = 4 |

---

## Algorithms

### TSP

| Algorithm | Key parameters |
|---|---|
| **SA** – Simulated Annealing | T₀, α (cooling), steps_per_temp |
| **GA** – Genetic Algorithm | pop, crossover, mutation_rate, tournament_k |
| **HC** – Hill Climbing | mode (best / first improvement) |
| **TLBO** – Teaching–Learning-Based Optimization | pop_size, move_frac_max |
| **ACO** – Ant Colony Optimization | n_ants, α (pheromone), β (heuristic), ρ (evaporation) |

### Knapsack

| Algorithm | Key parameters |
|---|---|
| **SA_KP** | T₀, α, steps_per_temp |
| **GA_KP** | population_size, crossover_rate, mutation_rate, tournament_k |
| **HC_KP** | mode (best / first) |
| **TLBO_KP** | pop_size, move_frac_max |
| **ABC_KP** – Artificial Bee Colony | sn (scout bees), limit |

### Graph Coloring

| Algorithm | Key parameters |
|---|---|
| **SA_GC** | T₀, α |
| **GA_GC** | population_size, crossover_rate, mutation_rate, tournament_k |
| **HC_GC** | mode (best / first) |
| **ACO_GC** | n_ants, α (pheromone), β (heuristic), ρ (evaporation) |
| **DFS_GC** – Backtracking (exact) | time_limit_sec, max_backtracks |

---

## Project Structure

```
.
├── configs/                        # YAML configs (reference defaults)
│   ├── config_tsp.yaml
│   ├── config_knapsack.yaml
│   └── config_graphcolor.yaml
│
├── src/
│   ├── problems/                   # Problem definitions
│   │   ├── tsp.py                  # TSPProblem (Euclidean, random instances)
│   │   ├── knapsack.py             # KnapsackProblem (0/1, penalty/repair)
│   │   └── graph_coloring.py       # GraphColoringProblem (k-coloring, conflict min)
│   │
│   ├── solvers/                    # Algorithm implementations
│   │   ├── tsp/                    # sa_tsp, ga_tsp, hc_tsp, tlbo_tsp, aco_tsp
│   │   ├── knapsack/               # sa_knapsack, ga_knapsack, hc_knapsack,
│   │   │                           #   tlbo_knapsack, abc_knapsack
│   │   └── graphcoloring/          # sa_graphcolor, ga_graphcolor, hc_graphcolor,
│   │                               #   aco_graphcolor, dfs_graphcolor
│   │
│   ├── experiments/                # Experiment runners (read YAML, write CSV)
│   │   ├── run_tsp.py
│   │   ├── run_knapsack.py
│   │   └── run_graph_coloring.py
│   │
│   ├── core/                       # Shared infrastructure
│   │   ├── base.py                 # RunResult dataclass
│   │   ├── config.py               # YAML loader
│   │   ├── logging.py              # RunLogger (run + trace CSVs)
│   │   ├── rng.py                  # Reproducible RNG helpers
│   │   ├── eval_norm_tsp.py        # Normalized eval budget for TSP
│   │   ├── eval_norm_knapsack.py   # Normalized eval budget for Knapsack
│   │   └── eval_norm_gc.py         # Normalized eval budget for Graph Coloring
│   │
│   ├── base_implement/             # Standalone generic algorithm templates
│   │   ├── simulated_annealing.py
│   │   ├── hill_climbing.py
│   │   ├── tlbo.py
│   │   ├── bfs.py
│   │   ├── dfs.py
│   │   └── astar.py
│   │
│   ├── parameter sensitivity/      # CSV outputs from sweeps
│   │   ├── TSP/{SA,GA,HC,TLBO,ACO}/
│   │   ├── Knapsack/{SA,GA,HC,TLBO,ABC}/
│   │   └── GraphColoring/{SA,GA,HC,ACO,DFS}/
│   │
│   └── plots/Discreate/            # Generated PDFs and GIFs
│       ├── TSP/
│       ├── Knapsack/
│       └── GraphColoring/
│
├── run_param_sensitivity.py        # Run all parameter sweep experiments
├── plot_param_sensitivity.py       # Plot per-parameter convergence curves (PDF)
├── plot_convergence_comparison.py  # Compare all algorithms, best config (PDF)
└── plot_convergence_comparasion_gif.py  # Animated convergence GIFs
```

---

## Normalized Evaluation Budget

Raw `evals_cost` counts objective queries, but each query has different computational cost across algorithms. A normalization factor converts raw evals to a common unit:

### TSP — reference: O(n) full tour evaluation

| Algorithm | Cost per eval | work_factor |
|---|---|---|
| HC, SA | O(1) delta via 2-opt | 1/n |
| GA, TLBO | O(n) full tour sum | 1.0 |
| ACO | O(n²) construction + score | n |

### Knapsack — reference: O(n) full evaluation

| Algorithm | Cost per eval | work_factor |
|---|---|---|
| HC_KP, SA_KP | O(1) single-bit flip delta | 1/n |
| GA_KP, TLBO_KP, ABC_KP | O(n) full sum | 1.0 |

### Graph Coloring — reference: O(|E|) full conflict count

| Algorithm | Cost per eval | work_factor |
|---|---|---|
| HC_GC, SA_GC | O(deg(v)) delta recolor | 2/n |
| GA_GC, ACO_GC, DFS_GC | O(|E|) full evaluate | 1.0 |

---

## Usage

### 1. Run parameter sensitivity experiments

```bash
# All three problems
.venv/bin/python3 run_param_sensitivity.py

# One problem
.venv/bin/python3 run_param_sensitivity.py --problem tsp
.venv/bin/python3 run_param_sensitivity.py --problem knapsack
.venv/bin/python3 run_param_sensitivity.py --problem graphcolor

# Filter by algorithm
.venv/bin/python3 run_param_sensitivity.py --problem tsp --algo SA GA
.venv/bin/python3 run_param_sensitivity.py --problem knapsack --algo TLBO ABC
.venv/bin/python3 run_param_sensitivity.py --problem graphcolor --algo SA ACO DFS

# Re-run even if outputs exist
.venv/bin/python3 run_param_sensitivity.py --problem graphcolor --force
```

**Available `--algo` values per problem:**

| Problem | Algorithms |
|---|---|
| TSP | `SA` `GA` `HC` `TLBO` `ACO` |
| Knapsack | `SA` `GA` `HC` `TLBO` `ABC` |
| GraphColoring | `SA` `GA` `HC` `ACO` `DFS` |

Outputs are written to `src/parameter sensitivity/{Problem}/{Algo}/{param}/` as `*_run.csv` and `*_trace.csv`.

---

### 2. Plot per-parameter sensitivity curves

```bash
# All experiments for a problem
.venv/bin/python3 plot_param_sensitivity.py --problem TSP
.venv/bin/python3 plot_param_sensitivity.py --problem Knapsack
.venv/bin/python3 plot_param_sensitivity.py --problem GraphColoring

# Specific experiment keys only
.venv/bin/python3 plot_param_sensitivity.py --problem TSP --exp SA_T0 GA_mutation
.venv/bin/python3 plot_param_sensitivity.py --problem GraphColoring --exp SA_alpha ACO_n_ants DFS_time_limit
```

Each experiment produces two PDFs (raw evals + normalized evals) in `src/plots/Discreate/{Problem}/`.

**Available `--exp` keys:**

| Problem | Keys |
|---|---|
| TSP | `SA_T0` `SA_steps_per_temp` `GA_pop` `GA_crossover` `GA_mutation` `GA_tournament_k` `HC_mode` `TLBO_pop` `TLBO_move_frac_max` `ACO_n_ants` `ACO_alpha` `ACO_beta` `ACO_rho` |
| Knapsack | `SA_T0` `SA_steps_per_temp` `GA_pop` `GA_crossover` `GA_mutation` `GA_tournament_k` `HC_mode` `TLBO_pop` `TLBO_move_frac_max` `ABC_sn` `ABC_limit` |
| GraphColoring | `SA_T0` `SA_alpha` `GA_pop` `GA_crossover` `GA_mutation` `GA_tournament_k` `HC_mode` `ACO_n_ants` `ACO_alpha` `ACO_beta` `ACO_rho` `DFS_time_limit` `DFS_max_backtracks` |

---

### 3. Compare all algorithms (best config per algo)

Generates one raw + one normalized PDF per problem.

```bash
# All problems
.venv/bin/python3 plot_convergence_comparison.py

# Specific problem(s)
.venv/bin/python3 plot_convergence_comparison.py --problem tsp
.venv/bin/python3 plot_convergence_comparison.py --problem knapsack
.venv/bin/python3 plot_convergence_comparison.py --problem graphcoloring
```

**Outputs:**

| Problem | Files |
|---|---|
| TSP | `src/plots/Discreate/TSP/convergence_comparison_raw.pdf` |
| | `src/plots/Discreate/TSP/convergence_comparison_normalized.pdf` |
| Knapsack | `src/plots/Discreate/Knapsack/kp_comparison_raw.pdf` |
| | `src/plots/Discreate/Knapsack/kp_comparison_normalized.pdf` |
| GraphColoring | `src/plots/Discreate/GraphColoring/gc_comparison_raw.pdf` |
| | `src/plots/Discreate/GraphColoring/gc_comparison_normalized.pdf` |

Config selection:
- **TSP**: hardcoded best config per algorithm (based on parameter sensitivity results)
- **Knapsack / GraphColoring**: auto-selected by scanning all `*_run.csv` files and picking the config with the best median final cost

---

### 4. Animated convergence GIFs

Same logic as the PDF comparison script, but produces animated GIFs (70 frames, 14 fps).

```bash
# All problems
.venv/bin/python3 plot_convergence_comparasion_gif.py

# Specific problem(s)
.venv/bin/python3 plot_convergence_comparasion_gif.py --problem tsp
.venv/bin/python3 plot_convergence_comparasion_gif.py --problem knapsack
.venv/bin/python3 plot_convergence_comparasion_gif.py --problem graphcoloring
```

**Outputs:** same paths as above but `.gif` instead of `.pdf`.

---

## CSV Output Format

### `*_run.csv` — one row per run

| Column | Description |
|---|---|
| `run_id` | Unique identifier (algo, instance, run) |
| `best_cost` | Final best objective value |
| `evals_cost` | Total objective evaluations used |
| `time_sec` | Wall time |
| `param_tag` | Parameter variant label |

### `*_trace.csv` — convergence trace per run

| Column | Description |
|---|---|
| `run_id` | Matches `*_run.csv` |
| `iter` | Iteration number |
| `best_cost` | Best cost found so far at this iteration |
| `evals_cost` | Cumulative evaluations at this point |
| `normalized_evals` | `evals_cost × work_factor` (where available) |

---

## Requirements

```
numpy
pandas
matplotlib
pyyaml
Pillow   # for GIF generation
```

Install with:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas matplotlib pyyaml Pillow
```

---

## Base Implementations

`src/base_implement/` contains standalone, dependency-free generic templates:

- `simulated_annealing.py` — requires `neighbor_fn`, `cost_fn`
- `hill_climbing.py` — requires `neighbors_fn`, `cost_fn`
- `tlbo.py` — requires `cost_fn`
- `bfs.py` / `dfs.py` — requires `neighbors_fn`, `goal_test`
- `astar.py` — requires `neighbors_fn`, `goal_test`, `heuristic`, optional `edge_cost`
