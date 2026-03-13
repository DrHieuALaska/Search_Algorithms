import numpy as np

def TLBO_with_evals(
    objective_function,
    DIMENSION,
    BOUNDS,
    POP_SIZE=50,
    MAX_ITERATIONS=500,
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
    
    # --- Initialize population ---

    lower = np.array([b[0] for b in BOUNDS])
    upper = np.array([b[1] for b in BOUNDS])

    if lower.shape[0] != DIMENSION or upper.shape[0] != DIMENSION:
        raise ValueError("Bounds must match the specified dimension.")

    population = np.random.uniform(lower, upper, (POP_SIZE, DIMENSION))
    fitness = np.array([counted_objective(ind) for ind in population])

    for _ in range(MAX_ITERATIONS):

        # -----------------
        # Teacher Phase (student learns from the teacher)
        # -----------------
        teacher_idx = np.argmin(fitness)
        teacher = population[teacher_idx]

        mean = np.mean(population, axis=0)

        TF = np.random.randint(1, 3)      # Teaching factor (randomly 1 or 2)

        for i in range(POP_SIZE):

            r = np.random.rand(DIMENSION)

            new_solution = population[i] + r * (teacher - TF * mean)

            new_solution = np.clip(new_solution, lower, upper)

            new_fitness = counted_objective(new_solution)

            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        # -----------------
        # Learner Phase (students learn from each other)
        # -----------------
        for i in range(POP_SIZE):

            j = np.random.randint(0, POP_SIZE)
            while j == i:
                j = np.random.randint(0, POP_SIZE)

            Xi = population[i]
            Xj = population[j]

            if fitness[i] < fitness[j]:
                new_solution = Xi + np.random.rand(DIMENSION) * (Xi - Xj)
            else:
                new_solution = Xi + np.random.rand(DIMENSION) * (Xj - Xi)

            new_solution = np.clip(new_solution, lower, upper)

            new_fitness = counted_objective(new_solution)

            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
    
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
import os

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
def TLBO_run_trials_multi_func_to_csv(FUNCTIONS_FOR_TLBO, folder_path, file_name, n_trials=30):
    all_rows = []
    for fn_cfg in FUNCTIONS_FOR_TLBO:
        print(f"\n{'='*20}")
        print("Teaching-Learning-Based Optimization")
        print(f"\n{'='*55}")
        print(f"  {fn_cfg['name'].upper()} — {n_trials} trials")
        print(f"{'='*55}")

        kwargs = dict(
            objective_function = fn_cfg["func"],
            DIMENSION          = fn_cfg["dimension"],
            BOUNDS             = fn_cfg["bounds"],
            POP_SIZE           = fn_cfg["pop_size"],
            MAX_ITERATIONS     = fn_cfg["max_iter"],
            INTERVAL_EVALS     = fn_cfg["interval_evals"],
            F_TARGET           = fn_cfg["f_target"],
        )

        all_rows.extend(run_trials(fn_cfg["name"], TLBO_with_evals, kwargs, n_trials=n_trials))

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

