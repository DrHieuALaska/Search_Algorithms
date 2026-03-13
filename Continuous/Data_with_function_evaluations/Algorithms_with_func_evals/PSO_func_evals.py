import os

import numpy as np

def particle_swarm_optimization_with_evals(
    objective_function,
    DIMENSION,
    BOUNDS,
    NUM_PARTICLES=30,
    MAX_ITERATIONS=100,
    w=0.7,          # inertia
    c1=1.5,         # cognitive
    c2=1.5,          # social
    INTERVAL_EVALS=200,
    F_TARGET=None,      
):

    lower = np.array([b[0] for b in BOUNDS])
    upper = np.array([b[1] for b in BOUNDS])

    if lower.shape[0] != DIMENSION or upper.shape[0] != DIMENSION:
        raise ValueError("Bounds must match the specified dimension.")
    

    COUNT_evals = 0
    gbest_score = np.inf
    gbest_position = None
    convergence_history = []          # (eval_count, best_fitness) snapshots

    # --- Convergence tracking state ---
    convergence_evals = None                  

    def counted_objective(x):
        nonlocal COUNT_evals, gbest_score, gbest_position
        nonlocal convergence_evals

        val = objective_function(x)
        COUNT_evals += 1

        # Update best
        if val < gbest_score:
            gbest_score = val
            gbest_position = x.copy()

        # --- Strategy 1: First-Hitting Time ---
        if convergence_evals is None and F_TARGET is not None:
            if gbest_score <= F_TARGET:
                convergence_evals = COUNT_evals

        # Snapshot for convergence curve
        if COUNT_evals % INTERVAL_EVALS == 0:
            convergence_history.append((COUNT_evals, gbest_score))

        return val


    # Initialize particles
    positions = np.random.uniform(lower, upper, (NUM_PARTICLES, DIMENSION))
    velocities = np.zeros((NUM_PARTICLES, DIMENSION))

    # Personal best
    pbest_positions = positions.copy()
    pbest_scores = np.array([counted_objective(p) for p in positions])


    for _ in range(MAX_ITERATIONS):

        r1 = np.random.rand(NUM_PARTICLES, DIMENSION)
        r2 = np.random.rand(NUM_PARTICLES, DIMENSION)

        # Update velocity
        velocities = (
            w * velocities
            + c1 * r1 * (pbest_positions - positions)
            + c2 * r2 * (gbest_position - positions)
        )

        # Update position
        positions = positions + velocities
        positions = np.clip(positions, lower, upper)

        # Evaluate
        scores = np.array([counted_objective(p) for p in positions])

        # Update personal best
        better_mask = scores < pbest_scores
        pbest_positions[better_mask] = positions[better_mask]
        pbest_scores[better_mask] = scores[better_mask]

    success = True if convergence_evals is not None else False
    if(convergence_evals is None):
        convergence_evals = COUNT_evals # penalty: if never hit target, set convergence evals to total evals

    return {
        "best_solution": gbest_position,
        "best_fitness": gbest_score,
        "success": success,
        "total_evals": COUNT_evals,
        "convergence_history": convergence_history,  # list of (eval, fitness)
        "convergence_evals": convergence_evals
    }


def PSO_run_trials_multi_func_to_csv(FUNCTIONS_FOR_PSO, file_name, n_trials=30):
    all_rows = []
    for fn_cfg in FUNCTIONS_FOR_PSO:
        print(f"\n{'='*20}")
        print("PSO")
        print(f"\n{'='*50}")
        print(f"  {fn_cfg['name'].upper()} — {n_trials} trials")
        print(f"{'='*55}")

        kwargs = dict(
            objective_function = fn_cfg["func"],
            DIMENSION          = fn_cfg["dimension"],
            BOUNDS             = fn_cfg["bounds"],
            NUM_PARTICLES      = fn_cfg["num_particles"],
            MAX_ITERATIONS     = fn_cfg["max_iter"],
            w                  = fn_cfg["w"],
            c1                 = fn_cfg["c1"],
            c2                 = fn_cfg["c2"],
            INTERVAL_EVALS     = fn_cfg["interval_evals"],
            F_TARGET           = fn_cfg["f_target"],
        )

        all_rows.extend(run_trials(fn_cfg["name"], particle_swarm_optimization_with_evals, kwargs, n_trials=n_trials))

    # Build DataFrame — missing f-columns become NaN
    fixed_cols = ["func_name", "best_fitness", "convergence_evals", "total_evals", "success"]
    f_cols = sorted(
        [c for c in all_rows[0].keys() if c.startswith("f") and c[1:].isdigit()],
        key=lambda x: int(x[1:])
    )
    df = pd.DataFrame(all_rows)[fixed_cols + f_cols]

    # Save
    df.to_csv(file_name, index=False)
    print(f"\nSaved {file_name} — {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Convergence history columns: f1 → {f_cols[-1]}")

import pandas as pd

# -----------------------------------------------
# Multi-trial runner (returns list of row dicts)
# -----------------------------------------------
def run_trials(func_name, algo_func, algo_kwargs, n_trials=30):
    rows = []
    for trial in range(n_trials):
        np.random.seed(trial)
        r = algo_func(**algo_kwargs)

        row = {
            "func_name":         func_name,
            "best_fitness":      r["best_fitness"],
            "convergence_evals": r["convergence_evals"],
            "total_evals":       r["total_evals"],
            "success":           r["success"],
        }
        for (evals, fit) in (r["convergence_history"]):
            row[f"f{evals}"] = fit

        rows.append(row)
    return rows


def PSO_run_trials_multi_func_to_csv(FUNCTIONS_FOR_PSO, folder_path, file_name, n_trials=30):    
    all_rows = []
    for fn_cfg in FUNCTIONS_FOR_PSO:
        print(f"\n{'='*20}")
        print("PSO")
        print(f"\n{'='*50}")
        print(f"  {fn_cfg['name'].upper()} — {n_trials} trials")
        print(f"{'='*55}")

        kwargs = dict(
            objective_function = fn_cfg["func"],
            DIMENSION          = fn_cfg["dimension"],
            BOUNDS             = fn_cfg["bounds"],
            NUM_PARTICLES      = fn_cfg["num_particles"],
            MAX_ITERATIONS     = fn_cfg["max_iter"],
            w                  = fn_cfg["w"],
            c1                 = fn_cfg["c1"],
            c2                 = fn_cfg["c2"],
            INTERVAL_EVALS     = fn_cfg["interval_evals"],
            F_TARGET           = fn_cfg["f_target"],
        )

        all_rows.extend(run_trials(fn_cfg["name"], particle_swarm_optimization_with_evals, kwargs, n_trials=n_trials))

    # Build DataFrame — missing f-columns become NaN
    fixed_cols = ["func_name", "best_fitness", "convergence_evals", "total_evals", "success"]
    f_cols = sorted(
        [c for c in all_rows[0].keys() if c.startswith("f") and c[1:].isdigit()],
        key=lambda x: int(x[1:])
    )
    df = pd.DataFrame(all_rows)[fixed_cols + f_cols]

    # Save
    df.to_csv(os.path.join(folder_path, file_name), index=False)
    print(f"\nSaved {file_name} — {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Convergence history columns: f1 → {f_cols[-1]}")
