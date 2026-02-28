import os

import numpy as np

def artificial_bee_colony_with_evals(
    objective_function,
    DIMENSION,
    BOUNDS,
    COLONY_SIZE=20,
    MAX_ITERATIONS=100,
    LIMIT=20,
    PHI_RANGE=(-1, 1),
    # --- Convergence tracking params ---
    INTERVAL_EVALS=100,
    F_TARGET=None,      
    # tolerance for close enough to best fitness to be considered convergence
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

    # -------------------------
    # Bounds
    # -------------------------
    LOWER_BOUND = np.array([b[0] for b in BOUNDS])
    UPPER_BOUND = np.array([b[1] for b in BOUNDS])

    # -------------------------
    # Initialization
    # -------------------------
    NUM_FOOD_SOURCES = COLONY_SIZE // 2
    food_sources = np.random.uniform(LOWER_BOUND, UPPER_BOUND, (NUM_FOOD_SOURCES, DIMENSION))
    fitness = np.array([counted_objective(x) for x in food_sources])
    trial_counter = np.zeros(NUM_FOOD_SOURCES)

    def calculate_probabilities(fitness):
        inv_fit = 1 / (1 + fitness - np.min(fitness))
        return inv_fit / np.sum(inv_fit)

    # -------------------------
    # Main Loop
    # -------------------------
    for _ in range(MAX_ITERATIONS):
        # Employed Bee Phase
        for i in range(NUM_FOOD_SOURCES):
            k = np.random.choice([j for j in range(NUM_FOOD_SOURCES) if j != i])
            phi = np.random.uniform(PHI_RANGE[0], PHI_RANGE[1], DIMENSION)
            candidate = food_sources[i] + phi * (food_sources[i] - food_sources[k])
            candidate = np.clip(candidate, LOWER_BOUND, UPPER_BOUND)
            candidate_fitness = counted_objective(candidate)
            if candidate_fitness < fitness[i]:
                food_sources[i] = candidate
                fitness[i] = candidate_fitness
                trial_counter[i] = 0
            else:
                trial_counter[i] += 1

        # Onlooker Bee Phase
        probabilities = calculate_probabilities(fitness)
        for _ in range(NUM_FOOD_SOURCES):
            i = np.random.choice(NUM_FOOD_SOURCES, p=probabilities)
            k = np.random.choice([j for j in range(NUM_FOOD_SOURCES) if j != i])
            phi = np.random.uniform(PHI_RANGE[0], PHI_RANGE[1], DIMENSION)
            candidate = food_sources[i] + phi * (food_sources[i] - food_sources[k])
            candidate = np.clip(candidate, LOWER_BOUND, UPPER_BOUND)
            candidate_fitness = counted_objective(candidate)
            if candidate_fitness < fitness[i]:
                food_sources[i] = candidate
                fitness[i] = candidate_fitness
                trial_counter[i] = 0
            else:
                trial_counter[i] += 1

        # Scout Bee Phase
        for i in range(NUM_FOOD_SOURCES):
            if trial_counter[i] >= LIMIT:
                food_sources[i] = np.random.uniform(LOWER_BOUND, UPPER_BOUND)
                fitness[i] = counted_objective(food_sources[i])
                trial_counter[i] = 0

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
def ABC_run_trials_multi_func_to_csv(FUNCTIONS_FOR_ABC, folder_path, file_name, n_trials=30):
    all_rows = []
    for fn_cfg in FUNCTIONS_FOR_ABC:
        print(f"\n{'='*20}")
        print("ABC")
        print(f"\n{'='*55}")
        print(f"  {fn_cfg['name'].upper()} — {n_trials} trials")
        print(f"{'='*55}")

        kwargs = dict(
            objective_function = fn_cfg["func"],
            DIMENSION          = fn_cfg["dimension"],
            BOUNDS             = fn_cfg["bounds"],
            COLONY_SIZE        = fn_cfg["colony_size"],
            MAX_ITERATIONS     = fn_cfg["max_iter"],
            LIMIT              = fn_cfg["limit"],
            PHI_RANGE          = fn_cfg["phi_range"],
            INTERVAL_EVALS     = fn_cfg["interval_evals"],
            F_TARGET           = fn_cfg["f_target"],
        )

        all_rows.extend(run_trials(fn_cfg["name"], artificial_bee_colony_with_evals, kwargs, n_trials=n_trials))

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




























# summary = summarize_trials(results, total_evals_budget=TOTAL_BUDGET)

# # -----------------------------------------------
# # Plot median convergence curve across trials
# # -----------------------------------------------
# max_len = max(len(r["convergence_history"]) for r in results)
# padded = []
# for r in results:
#     hist = r["convergence_history"]
#     last_val = hist[-1][1] if hist else np.inf
#     # pad to same length with last known best
#     padded.append([v for _, v in hist] + [last_val] * (max_len - len(hist)))

# padded = np.array(padded)
# eval_axis = np.arange(1, max_len + 1) * algo_kwargs["INTERVAL_EVALS"]

# plt.figure(figsize=(9, 5))
# plt.plot(eval_axis, np.median(padded, axis=0), label="Median", linewidth=2)
# plt.fill_between(eval_axis,
#                  np.percentile(padded, 25, axis=0),
#                  np.percentile(padded, 75, axis=0),
#                  alpha=0.3, label="IQR (25–75%)")
# plt.axhline(algo_kwargs["F_TARGET"], color="red", linestyle="--", label=f"FHT target = {algo_kwargs['F_TARGET']}")
# plt.xlabel("Function Evaluations")
# plt.ylabel("Best Fitness")
# plt.title("ABC on Rastrigin (10D) — 20 Trials")
# plt.legend()
# plt.yscale("log")
# plt.tight_layout()
# plt.show()