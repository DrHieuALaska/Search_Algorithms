import numpy as np

def differential_evolution_with_evals(
    objective_function,
    DIMENSION,
    BOUNDS,
    POPULATION_SIZE=50,
    MAX_ITERATIONS=200,
    F=0.7,              # scaling factor
    CR=0.9,             # crossover rate
    INTERVAL_EVALS=200,
    F_TARGET=None     # target fitness for first-hitting-time
):
    
    #this version of DE counts the number of function evaluations
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
        
    # -------------------------
    # Bounds
    # -------------------------
    LOWER_BOUND = np.array([b[0] for b in BOUNDS])
    UPPER_BOUND = np.array([b[1] for b in BOUNDS])

    if LOWER_BOUND.shape[0] != DIMENSION or UPPER_BOUND.shape[0] != DIMENSION:
        raise ValueError("Bounds must match the specified dimension.")

    # -------------------------
    # Initialize population
    # -------------------------
    population = LOWER_BOUND + (UPPER_BOUND - LOWER_BOUND) * np.random.rand(POPULATION_SIZE, DIMENSION)
    fitness = np.array([counted_objective(x) for x in population])


    # -------------------------
    # Main loop
    # -------------------------
    for _ in range(MAX_ITERATIONS):

        for i in range(POPULATION_SIZE):

            # --- Mutation (DE/rand/1) ---
            idxs = list(range(POPULATION_SIZE))
            idxs.remove(i)
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

            mutant = population[r1] + F * (population[r2] - population[r3])

            # Bound handling
            mutant = np.clip(mutant, LOWER_BOUND, UPPER_BOUND)

            # --- Crossover (binomial) ---
            trial = population[i].copy()
            j_rand = np.random.randint(0, DIMENSION)  # ensure at least one dimension from mutant

            for j in range(DIMENSION):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]

            # --- Selection (greedy) ---
            trial_fitness = counted_objective(trial)

            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

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
import os

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


def DE_run_trials_multi_func_to_csv(FUNCTIONS_FOR_DE, folder_path, file_name, n_trials=30):
    all_rows = []
    for fn_cfg in FUNCTIONS_FOR_DE:
        print(f"\n{'='*20}")
        print("DIFFERENTIAL EVOLUTION")
        print(f"\n{'='*50}")
        print(f"  {fn_cfg['name'].upper()} — {n_trials} trials")
        print(f"{'='*55}")

        kwargs = dict(
            objective_function = fn_cfg["func"],
            DIMENSION          = fn_cfg["dimension"],
            BOUNDS             = fn_cfg["bounds"],
            POPULATION_SIZE    = fn_cfg["pop_size"],
            MAX_ITERATIONS     = fn_cfg["max_iter"],
            F                  = fn_cfg["F"],
            CR                 = fn_cfg["CR"],
            INTERVAL_EVALS     = fn_cfg["interval_evals"],
            F_TARGET           = fn_cfg["f_target"]
        )

        all_rows.extend(run_trials(fn_cfg["name"], differential_evolution_with_evals, kwargs, n_trials=n_trials))


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
