import os

import numpy as np

def firefly_algorithm_with_evals(
    objective_function,
    DIMENSION,
    BOUNDS,
    NUM_FIREFLIES=25,
    MAX_ITERATIONS=100,
    alpha=0.2,
    beta0=1.0,
    gamma=1.0,
    INTERVAL_EVALS=200,
    F_TARGET=None,
):
     # This version of the Firefly algorithm counts the number of function evaluations.
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


    # --- Initialize population ---
    lower = np.array([b[0] for b in BOUNDS])
    upper = np.array([b[1] for b in BOUNDS])

    if lower.shape[0] != DIMENSION or upper.shape[0] != DIMENSION:
        raise ValueError("Bounds must match the specified dimension.")

    # initialize fireflies uniform randomly within bounds
    fireflies = lower + (upper - lower) * np.random.rand(NUM_FIREFLIES, DIMENSION)


    brightness = np.array([counted_objective(f) for f in fireflies])


    # --- Main loop ---
    for _ in range(MAX_ITERATIONS):
        for i in range(NUM_FIREFLIES):
            for j in range(NUM_FIREFLIES):

                if brightness[j] < brightness[i]:

                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = beta0 * np.exp(-gamma * r**2) # attractiveness decreases with distance 

                    step = beta * (fireflies[j] - fireflies[i]) # move towards brighter firefly
                    random_step = alpha * (np.random.rand(DIMENSION) - 0.5) # random perturbation alpha * [-0.5, 0.5]

                    fireflies[i] += step + random_step

                    # Apply bounds
                    fireflies[i] = np.clip(fireflies[i], lower, upper)

                    brightness[i] = counted_objective(fireflies[i])

    success = True if convergence_evals is not None else False
    if(convergence_evals is None):
        convergence_evals = COUNT_evals # penalty: if never hit target, set convergence evals to total evals

    return {
        "best_solution": best_solution,
        "best_fitness": best_fitness,
        "success": success,
        "total_evals": COUNT_evals,
        "convergence_history": convergence_history,
        "convergence_evals": convergence_evals
    }



import pandas as pd

# -----------------------------------------------
# Multi-trial runner (returns list of row dicts)
# -----------------------------------------------
def run_trials(func_name, algo_func, algo_kwargs, n_trials=2):
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


def FireFly_run_trials_multi_func_to_csv(FUNCTIONS_FOR_FIREFLY, folder_path, file_name, n_trials=30):
    all_rows = []
    for fn_cfg in FUNCTIONS_FOR_FIREFLY:
        print(f"\n{'='*20}")
        print("FIREFLY")
        print(f"\n{'='*50}")
        print(f"  {fn_cfg['name'].upper()} — {n_trials} trials")
        print(f"{'='*55}")

        kwargs = dict(
            objective_function = fn_cfg["func"],
            DIMENSION          = fn_cfg["dimension"],
            BOUNDS             = fn_cfg["bounds"],
            NUM_FIREFLIES      = fn_cfg["num_fireflies"],
            MAX_ITERATIONS     = fn_cfg["max_iter"],
            alpha              = fn_cfg["alpha"],
            beta0              = fn_cfg["beta0"],
            gamma              = fn_cfg["gamma"],
            INTERVAL_EVALS     = fn_cfg["interval_evals"],
            F_TARGET           = fn_cfg["f_target"]
        )

        all_rows.extend(run_trials(fn_cfg["name"], firefly_algorithm_with_evals, kwargs, n_trials=n_trials))


    df = pd.DataFrame(all_rows)

    fixed_cols = ["func_name", "best_fitness", "convergence_evals", "total_evals", "success"]
    f_cols = sorted(
        [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
        key=lambda x: int(x[1:])
    )
    df = df[fixed_cols + f_cols]
    #save ro csv
    df.to_csv(os.path.join(folder_path, file_name), index=False)
    print(f"\nSaved {file_name} — {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Convergence history columns: f1 → {f_cols[-1]}")
