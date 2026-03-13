import pandas as pd
import numpy as np
import os

SOURCE_FOLDER = 'trials_data'
FOLDER_DISTRIBUTION_PATH = 'distribution_data'
FOLDER_GRAPH_PATH = 'graph_data'
# Create the directory if it doesn't exist
if not os.path.exists(FOLDER_DISTRIBUTION_PATH):
    os.makedirs(FOLDER_DISTRIBUTION_PATH)
if not os.path.exists(FOLDER_GRAPH_PATH):
    os.makedirs(FOLDER_GRAPH_PATH)


ALGORITHMS = ["abc", "cuckoo_search", "firefly", "pso", "de", "simulated_annealing", "hill_climbing", "ga", "tlbo"]

for algo in ALGORITHMS:
    df = pd.read_csv(os.path.join(SOURCE_FOLDER, f"{algo}_results.csv"))

    # identify f-columns (e.g. f200, f400, ...)
    f_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
     

    # -----------------------------------------------
    # Distribution CSV
    # -----------------------------------------------
    dist_rows = []

    for func_name, group in df.groupby("func_name"):

        row = {
            "func_name":               func_name,
            "success_rate":            group["success"].mean(),

            "best_fitness_mean":       group["best_fitness"].mean(),
            "best_fitness_std":        group["best_fitness"].std(),
            "best_fitness_median":     group["best_fitness"].median(),
            "best_fitness_best":       group["best_fitness"].min(),
            "best_fitness_worst":      group["best_fitness"].max(),

            # convergence_evals: failures penalized as total_evals
            "convergence_evals_mean":  group["convergence_evals"].mean(),
            "convergence_evals_std":   group["convergence_evals"].std(),
            "convergence_evals_median":group["convergence_evals"].median(),

            "total_evals_mean":        group["total_evals"].mean(),
        }
        dist_rows.append(row)

    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(os.path.join(FOLDER_DISTRIBUTION_PATH, f"{algo}_distribution.csv"), index=False)
    print(f"Saved {algo}_distribution.csv")

    # -----------------------------------------------
    # Graph CSV
    # -----------------------------------------------
    graph_rows = []

    for func_name, group in df.groupby("func_name"):
        row = {"func_name": func_name}
        filled = group[f_cols].ffill(axis=1)   # axis=1 = fill rightward within each trial row

        for col in f_cols:
            eval_num = col[1:]   # e.g. "200" from "f200"
            # forward-fill NaN: if a trial ended early, carry last known value
            vals = filled[col]
            row[f"f{eval_num}_mean"] = vals.mean()
            row[f"f{eval_num}_std"]  = vals.std()
            row[f"f{eval_num}_p50"]  = vals.median()
            row[f"f{eval_num}_p25"]  = vals.quantile(0.25)
            row[f"f{eval_num}_p75"]  = vals.quantile(0.75)
        graph_rows.append(row)

    graph_df = pd.DataFrame(graph_rows)
    graph_df.to_csv(os.path.join(FOLDER_GRAPH_PATH, f"{algo}_graph.csv"), index=False)
    print(f"Saved {algo}_graph.csv")
