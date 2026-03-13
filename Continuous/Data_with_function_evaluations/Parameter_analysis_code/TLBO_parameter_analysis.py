import numpy as np
import pandas as pd
import os
from Algorithms_with_func_evals.TLBO_func_evals import TLBO_with_evals

# -----------------------------------------------
# Default parameters (held fixed when another varies)
# -----------------------------------------------
DEFAULT_PARAMS = {
    "POP_SIZE": 20,
}


OUTPUT_FOLDER = "param_analysis_data"
OUTPUT_SUBFOLDER = "tlbo"
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_SUBFOLDER)
os.makedirs(OUTPUT_PATH, exist_ok=True)


def compute_max_iterations(pop_size, budget):

    init_evals      = pop_size
    evals_per_iter  = 2 * pop_size
    return max(1, int((budget - init_evals) // evals_per_iter))


def run_single_trial(fn_cfg, pop_size, interval_evals, budget, seed):
    max_iter = compute_max_iterations(pop_size, budget)
    np.random.seed(seed)
    result = TLBO_with_evals(
        objective_function = fn_cfg["func"],
        DIMENSION          = len(fn_cfg["bounds"]),
        BOUNDS             = fn_cfg["bounds"],
        POP_SIZE           = pop_size,
        MAX_ITERATIONS     = max_iter,
        INTERVAL_EVALS     = interval_evals,
        F_TARGET           = fn_cfg["f_target"],
    )
    return result


def TLBO_run_param_analysis(param_name, param_values, OBJECTIVE_FUNCTIONS, n_trials, budget):
    """
    Vary one parameter across param_values, hold others at DEFAULT_PARAMS.
    Saves one CSV per parameter.
    """
    all_rows = []
    interval_evals = budget // 500   # ~500 checkpoints per trial

    for param_value in param_values:
        # Build full param set: defaults + override the one being varied
        pop_size = DEFAULT_PARAMS["POP_SIZE"]

        if param_name == "pop_size": pop_size = param_value

        max_iter = compute_max_iterations(pop_size, budget)

        print(f"\n{'='*55}")
        print(f"  {param_name} = {param_value}  |  MAX_ITER = {max_iter}  |  POP_SIZE = {pop_size}")
        print(f"{'='*55}")

        for fn_cfg in OBJECTIVE_FUNCTIONS:
            print(f"  [{fn_cfg['name']}]", end="", flush=True)

            for trial in range(n_trials):
                result = run_single_trial(fn_cfg, pop_size, interval_evals, budget= budget, seed=trial)

                row = {
                    "param_name":        param_name,
                    "param_value":       str(param_value),   # str so tuples serialize cleanly
                    "func_name":         fn_cfg["name"],
                    "pop_size":        pop_size,
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

    fixed_cols = ["param_name", "param_value", "func_name", "pop_size",
                  "max_iterations", "trial",
                  "best_fitness", "convergence_evals", "total_evals", "success"]
    
    f_cols = sorted(
        [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
        key=lambda x: int(x[1:])
    )
    df = df[fixed_cols + f_cols]

    out_path = os.path.join(OUTPUT_PATH, f"tlbo_param_{param_name.lower()}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} — {df.shape[0]} rows × {df.shape[1]} cols")
    return df

