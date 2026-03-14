import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# -----------------------------------------------
# Constants
# -----------------------------------------------
COLORS = {
    "ABC":           "#2196F3",
    "Cuckoo Search": "#4CAF50",
    "Firefly":       "#FF5722",
    "PSO":           "#9C27B0",
    "DE":            "#FF9800",
    "Simulated Annealing": "#3F51B5",
    "Hill Climbing":      "#009688",
    "TLBO":               "#795548",
    "GA":                 "#E91E63",
}

DIM_STYLES = {
    "10D": {"linestyle": "-",  "alpha_fill": 0.18},
    "30D": {"linestyle": "--", "alpha_fill": 0.13},
    "50D": {"linestyle": ":",  "alpha_fill": 0.08},
}


# -----------------------------------------------
# Core data extractor
# -----------------------------------------------
def get_eval_axis_and_values(df, func_name, mode="median_iqr"):
    """
    Extract eval checkpoints and center/band arrays for a given function.

    Parameters
    ----------
    df        : DataFrame loaded from *_graph.csv
    func_name : e.g. "sphere", "rastrigin", "rosenbrock"
    mode      : "median_iqr"  → center=median, lower=p25, upper=p75
                "mean_std"    → center=mean,   lower=mean-std, upper=mean+std

    Returns
    -------
    eval_axis, center, lower, upper  — or  None, None, None, None if not found
    """
    row = df[df["func_name"] == func_name]
    if row.empty:
        return None, None, None, None

    if mode == "median_iqr":
        center_suffix = "_p50"
        lower_suffix  = "_p25"
        upper_suffix  = "_p75"
    elif mode == "mean_std":
        center_suffix = "_mean"
    else:
        raise ValueError(f"mode must be 'median_iqr' or 'mean_std', got '{mode}'")

    center_cols = sorted(
        [c for c in df.columns if c.endswith(center_suffix) and c[1:].split("_")[0].isdigit()],
        key=lambda x: int(x[1:].split("_")[0])
    )
    if not center_cols:
        return None, None, None, None

    eval_axis = np.array([int(c[1:].split("_")[0]) for c in center_cols])
    center    = row[center_cols].values.flatten().astype(float)

    if mode == "median_iqr":
        lower = row[[c.replace(center_suffix, lower_suffix) for c in center_cols]].values.flatten().astype(float)
        upper = row[[c.replace(center_suffix, upper_suffix) for c in center_cols]].values.flatten().astype(float)
    else:
        std   = row[[c.replace("_mean", "_std") for c in center_cols]].values.flatten().astype(float)
        lower = center - std
        upper = center + std

    lower = np.maximum(lower, 1e-10)
    return eval_axis, center, lower, upper


def _band_label(mode):
    return "median (IQR 25–75%)" if mode == "median_iqr" else "mean ± std"

def _y_label(mode):
    return "Best Fitness (median + IQR)" if mode == "median_iqr" else "Best Fitness (mean ± std)"


# -----------------------------------------------
# Plot 1: one figure per function, all algos overlaid
# -----------------------------------------------
def plot_per_function(data, functions, output_dir, mode="median_iqr"):
    os.makedirs(output_dir, exist_ok=True)

    for func_name in functions:
        fig, ax = plt.subplots(figsize=(9, 5))

        for algo_name, df in data.items():
            eval_axis, center, lower, upper = get_eval_axis_and_values(df, func_name, mode)
            if eval_axis is None:
                continue
            color = COLORS.get(algo_name, "black")
            ax.plot(eval_axis, center, label=algo_name, color=color, linewidth=2)
            ax.fill_between(eval_axis, lower, upper, alpha=0.15, color=color)

        ax.set_xlabel("Function Evaluations", fontsize=12)
        ax.set_ylabel(_y_label(mode), fontsize=12)
        ax.set_title(f"Convergence Comparison — {func_name.capitalize()}", fontsize=13)
        ax.legend(fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        fig.tight_layout()

        path = os.path.join(output_dir, f"comparison_{func_name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


# -----------------------------------------------
# Plot 2: one figure per algorithm, all functions as subplots
# -----------------------------------------------
def plot_per_algorithm(data, functions, output_dir, mode="median_iqr"):
    os.makedirs(output_dir, exist_ok=True)

    for algo_name, df in data.items():
        fig, axes = plt.subplots(1, len(functions), figsize=(6 * len(functions), 5))
        fig.suptitle(f"{algo_name} — Convergence by Function", fontsize=14)

        for ax, func_name in zip(axes, functions):
            eval_axis, center, lower, upper = get_eval_axis_and_values(df, func_name, mode)
            color = COLORS.get(algo_name, "black")

            if eval_axis is None:
                ax.set_title(f"{func_name} (no data)")
                continue

            ax.plot(eval_axis, center, color=color, linewidth=2)
            ax.fill_between(eval_axis, lower, upper, alpha=0.2, color=color)
            ax.set_title(func_name.capitalize(), fontsize=12)
            ax.set_xlabel("Function Evaluations", fontsize=10)
            ax.set_ylabel(_y_label(mode) if func_name == functions[0] else "", fontsize=10)
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)

        fig.tight_layout()
        safe_name = algo_name.replace(" ", "_").lower()
        path = os.path.join(output_dir, f"{safe_name}_all_funcs.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


# -----------------------------------------------
# Plot 3: full N_func × N_algo grid
# -----------------------------------------------
def plot_full_grid(data, functions, output_dir, mode="median_iqr"):
    os.makedirs(output_dir, exist_ok=True)

    algo_list = list(data.keys())
    n_rows    = len(functions)
    n_cols    = len(algo_list)

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.35)

    for row_idx, func_name in enumerate(functions):
        for col_idx, algo_name in enumerate(algo_list):
            ax    = fig.add_subplot(gs[row_idx, col_idx])
            df    = data[algo_name]
            color = COLORS.get(algo_name, "black")

            eval_axis, center, lower, upper = get_eval_axis_and_values(df, func_name, mode)

            if eval_axis is not None:
                ax.plot(eval_axis, center, color=color, linewidth=2)
                ax.fill_between(eval_axis, lower, upper, alpha=0.2, color=color)
                ax.set_yscale("log")
                ax.grid(True, which="both", linestyle="--", alpha=0.4)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)

            if row_idx == 0:
                ax.set_title(algo_name, fontsize=12, fontweight="bold", color=color)
            if col_idx == 0:
                ax.set_ylabel(func_name.capitalize(), fontsize=11)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Evaluations", fontsize=9)

    fig.suptitle(
        f"Full Convergence Comparison — All Algorithms × All Functions\n({_band_label(mode)})",
        fontsize=14, fontweight="bold", y=1.01
    )

    path = os.path.join(output_dir, "full_comparison_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# -----------------------------------------------
# Convenience: run all three plots at once
# -----------------------------------------------
def plot_all(data, functions, output_dir, mode="median_iqr"):
    print(f"\nPlotting mode: {_band_label(mode)}")
    plot_per_function(data,  functions, output_dir, mode)
    plot_per_algorithm(data, functions, output_dir, mode)
    plot_full_grid(data,     functions, output_dir, mode)
    print(f"\nAll plots saved to ./{output_dir}/")


# -----------------------------------------------
# Plot scalability: one PNG per algorithm
#   subplots  = one per function
#   3 lines per subplot = 10D / 30D / 50D
#   output    = {output_dir}/{safe_algo_name}_all_funcs.png
# -----------------------------------------------
def plot_scalability(
    source_folder,
    algorithms,
    functions,
    dimensions=("10D", "30D", "50D"),
    output_dir="scalability",
    mode="median_iqr",
):
    """
    Read CSVs from  source_folder/{dim}/{csv_file}  for every dimension,
    then produce one PNG per algorithm.

    Each PNG has one subplot per objective function.
    Each subplot overlays three convergence curves: 10D, 30D, 50D,
    all drawn in the algorithm's own colour but with distinct line styles.

    Output files follow the same naming convention as plot_per_algorithm:
        {output_dir}/{safe_algo_name}_all_funcs.png

    Parameters
    ----------
    source_folder : str   root folder containing the dimension sub-folders
                          e.g. "graph_data"  →  graph_data/10D/, graph_data/30D/, …
    algorithms    : dict  { algo_name: csv_filename }
    functions     : list  objective function name strings
    dimensions    : sequence of dimension labels matching the sub-folder names
    output_dir    : str   folder to write the PNGs into  (default: "scalability")
    mode          : "median_iqr" or "mean_std"
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load all CSVs up front: dim_data[dim][algo_name] = DataFrame | None
    dim_data = {}
    for dim in dimensions:
        dim_data[dim] = {}
        for algo_name, csv_file in algorithms.items():
            csv_path = os.path.join(source_folder, dim, csv_file)
            if os.path.exists(csv_path):
                dim_data[dim][algo_name] = pd.read_csv(csv_path)
            else:
                print(f"  [warn] missing: {csv_path}")
                dim_data[dim][algo_name] = None

    print(f"\n[Scalability] Plotting mode: {_band_label(mode)}")

    for algo_name in algorithms:
        color = COLORS.get(algo_name, "black")

        fig, axes = plt.subplots(
            1, len(functions),
            figsize=(6 * len(functions), 5),
            squeeze=False,
        )
        fig.suptitle(
            f"{algo_name} — Scalability Across Dimensions  ({_band_label(mode)})",
            fontsize=13, fontweight="bold",
        )

        for col_idx, func_name in enumerate(functions):
            ax = axes[0][col_idx]

            for dim in dimensions:
                df = dim_data[dim].get(algo_name)
                if df is None:
                    continue

                eval_axis, center, lower, upper = get_eval_axis_and_values(
                    df, func_name, mode
                )
                if eval_axis is None:
                    continue

                style = DIM_STYLES.get(dim, {"linestyle": "-", "alpha_fill": 0.15})
                ax.plot(
                    eval_axis, center,
                    color=color,
                    linewidth=2,
                    linestyle=style["linestyle"],
                    label=dim,
                )
                ax.fill_between(
                    eval_axis, lower, upper,
                    alpha=style["alpha_fill"],
                    color=color,
                )

            ax.set_title(func_name.capitalize(), fontsize=12)
            ax.set_xlabel("Function Evaluations", fontsize=10)
            ax.set_ylabel(_y_label(mode) if col_idx == 0 else "", fontsize=10)
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.legend(title="Dimension", fontsize=9)

        fig.tight_layout()
        safe_name = algo_name.replace(" ", "_").lower()
        out_path  = os.path.join(output_dir, f"{safe_name}_all_funcs.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    print(f"\n[Scalability] All plots saved to ./{output_dir}/")


# -----------------------------------------------
# Example usage
# -----------------------------------------------
if __name__ == "__main__":
    ALGORITHMS = {
        "ABC":                "abc_graph.csv",
        "Cuckoo Search":      "cuckoo_search_graph.csv",
        "Firefly":            "firefly_graph.csv",
        "PSO":                "pso_graph.csv",
        "DE":                 "de_graph.csv",
        "Simulated Annealing":"simulated_annealing_graph.csv",
        "Hill Climbing":      "hill_climbing_graph.csv",
        "TLBO":               "tlbo_graph.csv",
        "GA":                 "ga_graph.csv",
    }
    FUNCTIONS     = ["sphere", "rastrigin", "rosenbrock"]
    SOURCE_FOLDER = "graph_data"

    plot_scalability(
        source_folder = SOURCE_FOLDER,
        algorithms    = ALGORITHMS,
        functions     = FUNCTIONS,
        dimensions    = ("10D", "30D", "50D"),
        output_dir    = "scalability",
        mode          = "median_iqr",
    )