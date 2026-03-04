import numpy as np
import os

def hill_climbing_with_evals(
    objective_function,
    DIMENSION,
    BOUNDS,
    N_STARTS=20,
    MAX_ITERATIONS=1000,
    STEP_SIZE=0.1,
    INTERVAL_EVALS=100,
    F_TARGET=None,
):
    
    COUNT_evals = 0
    best_fitness = np.inf
    best_solution = None
    convergence_history = []          # (eval_count, best_fitness) snapshots

    # --- Convergence tracking state ---
    convergence_evals = None                  

    def counted_objective(x):
        nonlocal COUNT_evals, best_fitness, best_solution
        nonlocal convergence_evals

        val = objective_function(x)
        COUNT_evals += 1

        # Update best
        if val < best_fitness:
            best_fitness = val
            best_solution = x.copy()

        # --- Strategy 1: First-Hitting Time ---
        if convergence_evals is None and F_TARGET is not None:
            if best_fitness <= F_TARGET:
                convergence_evals = COUNT_evals

        # Snapshot for convergence curve
        if COUNT_evals % INTERVAL_EVALS == 0:
            convergence_history.append((COUNT_evals, best_fitness))

        return val


    LOWER_BOUND = np.array([b[0] for b in BOUNDS])
    UPPER_BOUND = np.array([b[1] for b in BOUNDS])
    if len(LOWER_BOUND) != DIMENSION or len(UPPER_BOUND) != DIMENSION:
        raise ValueError("Bounds length must match dimension")

    # ----------------------------------------------------------

    # Initialize random solution
    current_positions = np.random.uniform(LOWER_BOUND, UPPER_BOUND, (N_STARTS, DIMENSION))
    current_fitnesses = np.array([counted_objective(pos) for pos in current_positions])

    for _ in range(MAX_ITERATIONS):
        neighbors = current_positions + np.random.uniform(-STEP_SIZE, STEP_SIZE, (N_STARTS, DIMENSION))
        neighbors = np.clip(neighbors, LOWER_BOUND, UPPER_BOUND)
        neighbor_fitnesses = np.array([counted_objective(nei) for nei in neighbors])

        # Move to better neighbors (greedy)
        better_mask = neighbor_fitnesses < current_fitnesses

        current_positions[better_mask] = neighbors[better_mask]
        current_fitnesses[better_mask] = neighbor_fitnesses[better_mask]
        
    success = True if convergence_evals is not None else False
    if(convergence_evals is None):
        convergence_evals = COUNT_evals # penalty: if never hit target, set convergence evals to total evals
        
    return {
        "best_solution": best_solution,
        "best_fitness": best_fitness,
        "success": success,
        "total_evals": COUNT_evals,
        "convergence_history": convergence_history,  # list of (eval, fitness)
        "convergence_evals": convergence_evals
    }


import pandas as pd

# -----------------------------------------------
# Multi-trial runner (returns list of row dicts)
# -----------------------------------------------
def run_trials(func_name, algo_func, algo_kwargs, n_trials=30):
    rows = []
    for trial in range(n_trials):
        np.random.seed(trial) # for reproducibility
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

# -----------------------------------------------
# Multi-trial for multiple functions & save to CSV
# -----------------------------------------------
def Hill_climbing_run_trials_multi_func_to_csv(FUNCTIONS_FOR_HILL_CLIMBING, folder_path, file_name, n_trials=30):
    all_rows = []
    for fn_cfg in FUNCTIONS_FOR_HILL_CLIMBING:
        print(f"\n{'='*20}")
        print("Hill Climbing")
        print(f"\n{'='*55}")
        print(f"  {fn_cfg['name'].upper()} — {n_trials} trials")
        print(f"{'='*55}")

        kwargs = dict(
            objective_function = fn_cfg["func"],
            DIMENSION          = fn_cfg["dimension"],
            BOUNDS             = fn_cfg["bounds"],
            N_STARTS           = fn_cfg["n_starts"],
            MAX_ITERATIONS     = fn_cfg["max_iter"],
            STEP_SIZE          = fn_cfg["step_size"],
            INTERVAL_EVALS     = fn_cfg["interval_evals"],
            F_TARGET           = fn_cfg["f_target"],
        )

        all_rows.extend(run_trials(fn_cfg["name"], hill_climbing_with_evals, kwargs, n_trials=n_trials))

    # Build DataFrame — missing f-columns become NaN
    df = pd.DataFrame(all_rows)

    fixed_cols = ["func_name", "best_fitness", "convergence_evals", "total_evals", "success"]
    f_cols = sorted(
        [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
        key=lambda x: int(x[1:])
    )
    df = df[fixed_cols + f_cols]

    # Save
    df.to_csv(os.path.join(folder_path, file_name), index=False)
    print(f"\nSaved {file_name} — {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Convergence history columns: f1 → {f_cols[-1]}")

