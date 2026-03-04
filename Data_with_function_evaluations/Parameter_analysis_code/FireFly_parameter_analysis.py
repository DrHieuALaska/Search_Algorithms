import numpy as np
import pandas as pd
import os
from Algorithms_with_func_evals.FireFly_func_evals import firefly_algorithm_with_evals

# -----------------------------------------------
# Default parameters (held fixed when another varies)
# -----------------------------------------------
DEFAULT_PARAMS = {
    "NUM_FIREFLIES": 20,
    "alpha": 0.2,
    "beta": 1.0,
    "fraction_for_gamma": 0.5,
}


OUTPUT_FOLDER = "param_analysis_data"
OUTPUT_SUBFOLDER = "firefly"
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_SUBFOLDER)
os.makedirs(OUTPUT_PATH, exist_ok=True)


def compute_max_iterations(num_fireflies, budget):

     
    #Firefly evals per iteration ≈ N²/2 (expected, early stage)
    #Uses worst-case estimate to ensure we stay within budget.
    init_evals      = num_fireflies
    evals_per_iter  = (num_fireflies * num_fireflies) // 2
    return max(1, int((budget - init_evals) // evals_per_iter))


def run_single_trial(fn_cfg, num_fireflies, alpha, beta0, gamma, interval_evals, budget,seed):
    max_iter = compute_max_iterations(num_fireflies, budget)
    np.random.seed(seed)
    result = firefly_algorithm_with_evals(
        objective_function = fn_cfg["func"],
        DIMENSION          = len(fn_cfg["bounds"]),
        BOUNDS             = fn_cfg["bounds"],
        NUM_FIREFLIES      = num_fireflies,
        alpha              = alpha,
        beta0              = beta0,
        gamma              = gamma,
        MAX_ITERATIONS     = max_iter,
        INTERVAL_EVALS     = interval_evals,
        F_TARGET           = fn_cfg["f_target"],
    )
    return result


def suggest_gamma(bounds, fraction=0.5):
    """
    Sets gamma so fireflies can 'see' each other at fraction * domain_diagonal.
    fraction=1.0 → only interact when very close
    fraction=0.1 → interact across most of the space (more global)
    """
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    diagonal = np.sqrt(np.sum((upper - lower)**2))   # full diagonal of search space (maximum distance)
    char_dist = fraction * diagonal
    return 1.0 / (char_dist**2)

def Fireflies_run_param_analysis(param_name, param_values, OBJECTIVE_FUNCTIONS, n_trials, budget):
    """
    Vary one parameter across param_values, hold others at DEFAULT_PARAMS.
    Saves one CSV per parameter.
    """
    all_rows = []
    interval_evals = budget // 500   # ~500 checkpoints per trial

    for param_value in param_values:
        # Build full param set: defaults + override the one being varied
        num_fireflies = DEFAULT_PARAMS["NUM_FIREFLIES"]
        alpha   = DEFAULT_PARAMS["alpha"]
        beta0       = DEFAULT_PARAMS["beta"]
        fraction_for_gamma       = DEFAULT_PARAMS["fraction_for_gamma"]

        if param_name == "NUM_FIREFLIES": num_fireflies = param_value
        elif param_name == "alpha":  alpha   = param_value
        elif param_name == "beta0":  beta0   = param_value
        elif param_name == "fraction_for_gamma":  fraction_for_gamma   = param_value

        gamma = suggest_gamma(OBJECTIVE_FUNCTIONS[0]["bounds"], fraction_for_gamma)

        max_iter = compute_max_iterations(num_fireflies, budget)

        print(f"\n{'='*55}")
        print(f"  {param_name} = {param_value}  |  MAX_ITER = {max_iter}  |  NUM_FIREFLIES = {num_fireflies}")
        print(f"{'='*55}")

        for fn_cfg in OBJECTIVE_FUNCTIONS:
            print(f"  [{fn_cfg['name']}]", end="", flush=True)

            for trial in range(n_trials):
                result = run_single_trial(fn_cfg, num_fireflies, alpha, beta0, gamma, interval_evals, budget= budget, seed=trial)

                row = {
                    "param_name":        param_name,
                    "param_value":       str(param_value),   # str so tuples serialize cleanly
                    "func_name":         fn_cfg["name"],
                    "num_fireflies":        num_fireflies,
                    "alpha":             alpha,
                    "beta0":             beta0,
                    "fraction_for_gamma":fraction_for_gamma,
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

    fixed_cols = ["param_name", "param_value", "func_name", "num_fireflies", "alpha", "beta0", "fraction_for_gamma",
                  "max_iterations", "trial",
                  "best_fitness", "convergence_evals", "total_evals", "success"]
    
    f_cols = sorted(
        [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
        key=lambda x: int(x[1:])
    )
    df = df[fixed_cols + f_cols]

    out_path = os.path.join(OUTPUT_PATH, f"firefly_param_{param_name.lower()}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} — {df.shape[0]} rows × {df.shape[1]} cols")
    return df

