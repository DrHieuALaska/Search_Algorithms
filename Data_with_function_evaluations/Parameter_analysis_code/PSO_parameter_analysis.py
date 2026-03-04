import numpy as np
import pandas as pd
import os
from Algorithms_with_func_evals.PSO_func_evals import particle_swarm_optimization_with_evals

# -----------------------------------------------
# Default parameters (held fixed when another varies)
# -----------------------------------------------
DEFAULT_PARAMS = {
    "NUM_PARTICLES": 40,
    "w": 0.7,
    "c1": 1.5,
    "c2": 1.5,
}


OUTPUT_FOLDER = "param_analysis_data"
OUTPUT_SUBFOLDER = "pso"
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_SUBFOLDER)
os.makedirs(OUTPUT_PATH, exist_ok=True)


def compute_max_iterations(num_particles, budget):

    # evals: initialization (num_particles) + per iter (num_particles, approx)
    return max(1, (budget - num_particles) // num_particles)


def run_single_trial(fn_cfg, num_particles, w, c1, c2, interval_evals, budget,seed):
    max_iter = compute_max_iterations(num_particles, budget)
    np.random.seed(seed)
    result = particle_swarm_optimization_with_evals(
        objective_function = fn_cfg["func"],
        DIMENSION          = len(fn_cfg["bounds"]),
        BOUNDS             = fn_cfg["bounds"],
        NUM_PARTICLES      = num_particles,
        w                  = w,
        c1                 = c1,
        c2                 = c2,
        MAX_ITERATIONS     = max_iter,
        INTERVAL_EVALS     = interval_evals,
        F_TARGET           = fn_cfg["f_target"],
    )
    return result


def PSO_run_param_analysis(param_name, param_values, OBJECTIVE_FUNCTIONS, n_trials, budget):
    """
    Vary one parameter across param_values, hold others at DEFAULT_PARAMS.
    Saves one CSV per parameter.
    """
    all_rows = []
    interval_evals = budget // 500   # ~500 checkpoints per trial

    for param_value in param_values:
        # Build full param set: defaults + override the one being varied
        num_particles = DEFAULT_PARAMS["NUM_PARTICLES"]
        w             = DEFAULT_PARAMS["w"]
        c1            = DEFAULT_PARAMS["c1"]
        c2            = DEFAULT_PARAMS["c2"]


        if param_name == "NUM_PARTICLES": num_particles = param_value
        elif param_name == "w":  w   = param_value
        elif param_name == "c1":  c1   = param_value
        elif param_name == "c2":  c2   = param_value

        max_iter = compute_max_iterations(num_particles=num_particles, budget=budget)

        print(f"\n{'='*55}")
        print(f"  {param_name} = {param_value}  |  MAX_ITER = {max_iter}  |  NUM_PARTICLES = {num_particles}")
        print(f"{'='*55}")

        for fn_cfg in OBJECTIVE_FUNCTIONS:
            print(f"  [{fn_cfg['name']}]", end="", flush=True)

            for trial in range(n_trials):
                result = run_single_trial(fn_cfg, num_particles, w, c1, c2, interval_evals, budget= budget, seed=trial)

                row = {
                    "param_name":        param_name,
                    "param_value":       str(param_value),   # str so tuples serialize cleanly
                    "func_name":         fn_cfg["name"],
                    "num_particles":     num_particles,
                    "w":                 w,
                    "c1":                c1,
                    "c2":                c2,
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

    fixed_cols = ["param_name", "param_value", "func_name", "num_particles", "w", "c1", "c2",
                  "max_iterations", "trial",
                  "best_fitness", "convergence_evals", "total_evals", "success"]

    f_cols = sorted(
        [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
        key=lambda x: int(x[1:])
    )

    df = df[fixed_cols + f_cols]

    out_path = os.path.join(OUTPUT_PATH, f"pso_param_{param_name.lower()}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} — {df.shape[0]} rows × {df.shape[1]} cols")
    return df

