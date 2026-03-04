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
}


# -----------------------------------------------
# Core data extractor
# -----------------------------------------------
def get_eval_axis_and_values(df, func_name, mode="median_iqr"):
    """
    Extract eval checkpoints and center/band arrays for a given function.

    Parameters
    ----------
    df   : DataFrame loaded from *_graph.csv
    func_name : e.g. "sphere", "rastrigin", "rosenbrock"
    mode : "median_iqr"  → center=median, lower=p25, upper=p75
           "mean_std"    → center=mean,   lower=mean-std, upper=mean+std

    Returns
    -------
    eval_axis : np.array of eval counts  e.g. [200, 400, ...]
    center    : np.array of center values
    lower     : np.array of lower band values  (already clamped to 1e-10)
    upper     : np.array of upper band values
    None, None, None, None  if func_name not found
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
        lower_suffix  = None   # derived from mean - std
        upper_suffix  = None
    else:
        raise ValueError(f"mode must be 'median_iqr' or 'mean_std', got '{mode}'")

    # find all center columns and sort by eval number
    center_cols = sorted(
        [c for c in df.columns if c.endswith(center_suffix) and c[1:].split("_")[0].isdigit()],
        key=lambda x: int(x[1:].split("_")[0])
    )
    if not center_cols:
        return None, None, None, None

    eval_axis = np.array([int(c[1:].split("_")[0]) for c in center_cols])
    center    = row[center_cols].values.flatten().astype(float)

    if mode == "median_iqr":
        lower_cols = [c.replace(center_suffix, lower_suffix) for c in center_cols]
        upper_cols = [c.replace(center_suffix, upper_suffix) for c in center_cols]
        lower = row[lower_cols].values.flatten().astype(float)
        upper = row[upper_cols].values.flatten().astype(float)
    else:
        std_cols = [c.replace("_mean", "_std") for c in center_cols]
        std      = row[std_cols].values.flatten().astype(float)
        lower    = center - std
        upper    = center + std

    # clamp lower band to avoid log-scale issues (can not be negative or zero)
    lower = np.maximum(lower, 1e-10)
    # upper = np.maximum(upper, 1e-10)

    return eval_axis, center, lower, upper


def _band_label(mode):
    return "median (IQR 25–75%)" if mode == "median_iqr" else "mean ± std"

def _y_label(mode):
    return "Best Fitness (median + IQR)" if mode == "median_iqr" else "Best Fitness (mean ± std)"


# -----------------------------------------------
# Plot 1: one figure per function, all algos overlaid
# -----------------------------------------------
def plot_per_function(data, functions, output_dir, mode="median_iqr"):
    """
    One PNG per objective function, all algorithms overlaid.

    Parameters
    ----------
    data       : dict  { algo_name: DataFrame }
    functions  : list of function name strings
    output_dir : folder to save PNGs
    mode       : "median_iqr" or "mean_std"
    """
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
        ax.set_title(f"Convergence Comparison — {func_name.capitalize()} (10D)", fontsize=13)
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
    """
    One PNG per algorithm, with one subplot per objective function.

    Parameters
    ----------
    data       : dict  { algo_name: DataFrame }
    functions  : list of function name strings
    output_dir : folder to save PNGs
    mode       : "median_iqr" or "mean_std"
    """
    os.makedirs(output_dir, exist_ok=True)

    for algo_name, df in data.items():
        fig, axes = plt.subplots(1, len(functions), figsize=(6 * len(functions), 5))
        fig.suptitle(f"{algo_name} — Convergence by Function (10D)", fontsize=14)

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
    """
    One large PNG: rows = functions, columns = algorithms.

    Parameters
    ----------
    data       : dict  { algo_name: DataFrame }
    functions  : list of function name strings
    output_dir : folder to save PNGs
    mode       : "median_iqr" or "mean_std"
    """
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

    mode_label = _band_label(mode)
    fig.suptitle(
        f"Full Convergence Comparison — All Algorithms × All Functions (10D)\n({mode_label})",
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
    """
    Run all three plot types in one call.

    Parameters
    ----------
    data       : dict  { algo_name: DataFrame }
    functions  : list of function name strings
    output_dir : folder to save PNGs
    mode       : "median_iqr" or "mean_std"
    """
    print(f"\nPlotting mode: {_band_label(mode)}")
    plot_per_function(data,  functions, output_dir, mode)
    plot_per_algorithm(data, functions, output_dir, mode)
    plot_full_grid(data,     functions, output_dir, mode)
    print(f"\nAll plots saved to ./{output_dir}/")


# -----------------------------------------------
# Example usage (run this file directly to test)
# -----------------------------------------------
if __name__ == "__main__":
    ALGORITHMS = {
        "ABC":           "abc_graph.csv",
        "Cuckoo Search": "cuckoo_search_graph.csv",
        "Firefly":       "firefly_graph.csv",
        "PSO":           "pso_graph.csv",
        "DE":            "de_graph.csv",
        "Simulated Annealing": "simulated_annealing_graph.csv",
        "Hill Climbing":      "hill_climbing_graph.csv",
    }
    FUNCTIONS     = ["sphere", "rastrigin", "rosenbrock"]
    SOURCE_FOLDER = "graph_data"
    OUTPUT_DIR_MEAN_STD    = "plots_mean_std"
    OUTPUT_DIR_MEDIAN_IQR  = "plots_median_iqr"

    data = {
        algo_name: pd.read_csv(os.path.join(SOURCE_FOLDER, file_name))
        for algo_name, file_name in ALGORITHMS.items()
    }

    plot_all(data, FUNCTIONS, OUTPUT_DIR_MEAN_STD, mode="mean_std")
    plot_all(data, FUNCTIONS, OUTPUT_DIR_MEDIAN_IQR, mode="median_iqr")