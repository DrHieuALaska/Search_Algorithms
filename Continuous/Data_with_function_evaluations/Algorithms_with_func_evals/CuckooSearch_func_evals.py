import os

import numpy as np
import math

def cuckoo_search_with_evals(
    objective_function,
    DIMENSION,
    BOUNDS,
    N_NESTS=25,
    MAX_ITERATIONS=100,
    pa=0.25,
    alpha=0.01,
    INTERVAL_EVALS=10,
    F_TARGET=None,
):
    
    # This version of the Cuckoo Search algorithm counts the number of function evaluations.
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


    # set up bounds
    lower = np.array([b[0] for b in BOUNDS])
    upper = np.array([b[1] for b in BOUNDS])

    if lower.shape[0] != DIMENSION or upper.shape[0] != DIMENSION:
        raise ValueError("Bounds must match the specified dimension.")

    # Initialize nests
    nests = lower + (upper - lower) * np.random.rand(N_NESTS, DIMENSION)
    fitness = np.array([counted_objective(nest) for nest in nests])

    # Lévy flight generator
    def levy_flight(size):
        beta = 1.5
        sigma = (
            math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        u = np.random.randn(*size) * sigma
        v = np.random.randn(*size)
        step = u / np.abs(v) ** (1 / beta)

        return step

    for _ in range(MAX_ITERATIONS):

        # Generate new solutions via Lévy flights
        steps = levy_flight((N_NESTS, DIMENSION))
        new_nests = nests + alpha * steps * (nests - best_solution)

        # Apply bounds
        new_nests = np.clip(new_nests, lower, upper)

        new_fitness = np.array([counted_objective(nest) for nest in new_nests])

        # Greedy selection
        improved = new_fitness < fitness
        nests[improved] = new_nests[improved]
        fitness[improved] = new_fitness[improved]

        # Replace fraction of worst nests

        sorted_idx = np.argsort(fitness)  
        n_replace = int(pa * N_NESTS)
        worst_idx = sorted_idx[-n_replace:]

        random_nests = lower + (upper - lower) * np.random.rand(n_replace, DIMENSION)
        random_fitness = np.array([counted_objective(nest) for nest in random_nests])

        nests[worst_idx] = random_nests
        fitness[worst_idx] = random_fitness
    

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


# -----------------------------------------------
# Multi-trial runner
# -----------------------------------------------
def run_trials(algo_func, algo_kwargs, n_trials=20):
    results = []
    for trial in range(n_trials):
        np.random.seed(trial)
        r = algo_func(**algo_kwargs)
        results.append(r)
        print(f"  Trial {trial+1:2d} | best={r['best_fitness']:.4f} | "
              f"convergence_evals={r['convergence_evals']} | Total evals={r['total_evals']}")
    return results


import pandas as pd

# -----------------------------------------------
# Multi-trial runner (returns list of row dicts)
# -----------------------------------------------
def run_trials_csv(func_name, algo_func, algo_kwargs, n_trials=30):
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

def CuckooSearch_run_trials_multi_func_to_csv(FUNCTIONS_FOR_CUCKOOSEARCH, folder_path, file_name, n_trials=30):
    all_rows = []
    for fn_cfg in FUNCTIONS_FOR_CUCKOOSEARCH:
        print(f"\n{'='*20}")
        print("CUCKOO SEARCH")
        print(f"\n{'='*55}")
        print(f"  {fn_cfg['name'].upper()} — {n_trials} trials")
        print(f"{'='*55}")

        kwargs = dict(
            objective_function = fn_cfg["func"],
            DIMENSION          = fn_cfg["dimension"],
            BOUNDS             = fn_cfg["bounds"],
            N_NESTS            = fn_cfg["n_nests"],
            MAX_ITERATIONS     = fn_cfg["max_iter"],
            pa                 = fn_cfg["pa"],
            alpha              = fn_cfg["alpha"],
            INTERVAL_EVALS     = fn_cfg["interval_evals"],
            F_TARGET           = fn_cfg["f_target"]
        )

        all_rows.extend(run_trials_csv(fn_cfg["name"], cuckoo_search_with_evals, kwargs, n_trials=n_trials))

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

