import numpy as np
import pandas as pd
import os
from Algorithms_with_func_evals.ABC_func_evals import artificial_bee_colony_with_evals

# -----------------------------------------------
# Default parameters (held fixed when another varies)
# -----------------------------------------------
DEFAULT_PARAMS = {
    "COLONY_SIZE": 40,
    "LIMIT":       100,
    "PHI_RANGE":   (-1, 1),
}


OUTPUT_FOLDER = "param_analysis_data"
OUTPUT_SUBFOLDER = "abc"
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_SUBFOLDER)
os.makedirs(OUTPUT_PATH, exist_ok=True)


def compute_max_iterations(colony_size, budget):
    """Keep total evals ≈ budget by adjusting MAX_ITERATIONS."""
    num_food_sources = colony_size // 2
    # evals: initialization (num_food_sources) + per iter (colony_size, approx)
    return max(1, int((budget - num_food_sources) // colony_size))


def run_single_trial(fn_cfg, colony_size, limit, phi_range, interval_evals, budget,seed):
    max_iter = compute_max_iterations(colony_size, budget)
    np.random.seed(seed)
    result = artificial_bee_colony_with_evals(
        objective_function = fn_cfg["func"],
        DIMENSION          = len(fn_cfg["bounds"]),
        BOUNDS             = fn_cfg["bounds"],
        COLONY_SIZE        = colony_size,
        MAX_ITERATIONS     = max_iter,
        LIMIT              = limit,
        PHI_RANGE          = phi_range,
        INTERVAL_EVALS     = interval_evals,
        F_TARGET           = fn_cfg["f_target"],
    )
    return result


def ABC_run_param_analysis(param_name, param_values, OBJECTIVE_FUNCTIONS, n_trials, budget):
    """
    Vary one parameter across param_values, hold others at DEFAULT_PARAMS.
    Saves one CSV per parameter.
    """
    all_rows = []
    interval_evals = budget // 500   # ~500 checkpoints per trial

    for param_value in param_values:
        # Build full param set: defaults + override the one being varied
        colony_size = DEFAULT_PARAMS["COLONY_SIZE"]
        limit       = DEFAULT_PARAMS["LIMIT"]
        phi_range   = DEFAULT_PARAMS["PHI_RANGE"]

        if param_name == "COLONY_SIZE": colony_size = param_value
        elif param_name == "LIMIT":     limit       = param_value
        elif param_name == "PHI_RANGE": phi_range   = param_value

        max_iter = compute_max_iterations(colony_size, budget)

        print(f"\n{'='*55}")
        print(f"  {param_name} = {param_value}  |  MAX_ITER = {max_iter}  |  COLONY = {colony_size}")
        print(f"{'='*55}")

        for fn_cfg in OBJECTIVE_FUNCTIONS:
            print(f"  [{fn_cfg['name']}]", end="", flush=True)

            for trial in range(n_trials):
                result = run_single_trial(fn_cfg, colony_size, limit, phi_range, interval_evals, budget= budget, seed=trial)

                row = {
                    "param_name":        param_name,
                    "param_value":       str(param_value),   # str so tuples serialize cleanly
                    "func_name":         fn_cfg["name"],
                    "colony_size":       colony_size,
                    "limit":             limit,
                    "phi_range":         str(phi_range),
                    "max_iterations":    max_iter,
                    "trial":             trial + 1,
                    "best_fitness":      result["best_fitness"],
                    "convergence_evals": result["convergence_evals"],
                    "total_evals":       result["total_evals"],
                    "success":           result["success"],
                }

                for (evals, fit) in (result["convergence_history"]):
                    row[f"f{evals}"] = fit

                all_rows.append(row)
                print(".", end="", flush=True)   # progress dot per trial

            print()   # newline after each function

    # -----------------------------------------------
    # Save
    # -----------------------------------------------
    df = pd.DataFrame(all_rows)

    fixed_cols = ["param_name", "param_value", "func_name", "colony_size",
                  "limit", "phi_range", "max_iterations", "trial",
                  "best_fitness", "convergence_evals", "total_evals", "success"]
    
    f_cols = sorted(
        [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
        key=lambda x: int(x[1:])
    )
    df = df[fixed_cols + f_cols]

    out_path = os.path.join(OUTPUT_PATH, f"abc_param_{param_name.lower()}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} — {df.shape[0]} rows × {df.shape[1]} cols")
    return df

