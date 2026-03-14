import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


# -----------------------------------------------
# Config
# -----------------------------------------------
FUNCTIONS = ["sphere", "rastrigin", "rosenbrock"]

# One color per param_value (auto-assigned from colormap)
def get_colors(n, cmap_name="tab10"):
    cmap = cm.get_cmap(cmap_name)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


# -----------------------------------------------
# Core: extract f-cols and compute median/IQR per param_value
# -----------------------------------------------
def compute_stats(df, func_name):
    """
    For a given function, group by param_value and compute
    median, p25, p75 across trials for each f-checkpoint.

    Returns dict: { param_value: (eval_axis, median, p25, p75) }
    """
    sub = df[df["func_name"] == func_name]
    if sub.empty:
        return {}

    f_cols = sorted(
        [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
        key=lambda x: int(x[1:])
    )
    if not f_cols:
        return {}

    eval_axis = np.array([int(c[1:]) for c in f_cols])

    stats = {}
    for param_value, group in sub.groupby("param_value"):
        # ffill across columns (time axis) per trial row
        filled = group[f_cols].ffill(axis=1)

        # if still NaN at f1 (trial had no history), fill with best_fitness
        for i, col in enumerate(f_cols):
            mask = filled[col].isna()
            if mask.any():
                filled.loc[mask, col] = group.loc[mask, "best_fitness"]

        median = filled.median(axis=0).values
        p25    = filled.quantile(0.25, axis=0).values
        p75    = filled.quantile(0.75, axis=0).values

        # clamp for log scale
        median = np.maximum(median, 1e-10)
        p25    = np.maximum(p25,    1e-10)
        p75    = np.maximum(p75,    1e-10)

        stats[param_value] = (eval_axis, median, p25, p75)

    return stats


# -----------------------------------------------
# Plot 1: one figure per CSV file
#   rows = functions, cols = 1
#   each line = one param_value
# -----------------------------------------------
def plot_param_file(csv_path, output_dir, functions=FUNCTIONS):
    """
    Reads one abc_param_<PARAM_NAME>.csv and produces:
      - one PNG with subplots for each function
      - one PNG per function (standalone)
    """
    df         = pd.read_csv(csv_path)
    param_name = df["param_name"].iloc[0]
    param_values = list(df["param_value"].unique())   # preserves CSV order
    colors       = get_colors(len(param_values))

    os.makedirs(output_dir, exist_ok=True)

    # --- multi-function subplot figure ---
    n_funcs = len(functions)
    fig, axes = plt.subplots(1, n_funcs, figsize=(6 * n_funcs, 5))
    if n_funcs == 1:
        axes = [axes]

    fig.suptitle(f"Parameter Analysis: {param_name}", fontsize=14, fontweight="bold")

    for ax, func_name in zip(axes, functions):
        stats = compute_stats(df, func_name)
        if not stats:
            ax.set_title(f"{func_name} (no data)")
            continue

        for color, param_value in zip(colors, param_values):
            if param_value not in stats:
                continue
            eval_axis, median, p25, p75 = stats[param_value]
            ax.plot(eval_axis, median, label=str(param_value), color=color, linewidth=2)
            ax.fill_between(eval_axis, p25, p75, alpha=0.15, color=color)

        ax.set_title(func_name.capitalize(), fontsize=12)
        ax.set_xlabel("Function Evaluations", fontsize=10)
        ax.set_ylabel("Best Fitness (median + IQR)" if func_name == functions[0] else "", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(title=param_name, fontsize=8, title_fontsize=9)

    fig.tight_layout()
    combined_path = os.path.join(output_dir, f"param_{param_name.lower()}_all_funcs.png")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {combined_path}")

    # --- one standalone PNG per function ---
    for func_name in functions:
        stats = compute_stats(df, func_name)
        if not stats:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for color, param_value in zip(colors, param_values):
            if param_value not in stats:
                continue
            eval_axis, median, p25, p75 = stats[param_value]
            ax.plot(eval_axis, median, label=str(param_value), color=color, linewidth=2)
            ax.fill_between(eval_axis, p25, p75, alpha=0.15, color=color)

        ax.set_title(f"{param_name} sensitivity — {func_name.capitalize()} (10D)", fontsize=13)
        ax.set_xlabel("Function Evaluations", fontsize=12)
        ax.set_ylabel("Best Fitness (median + IQR)", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(title=param_name, fontsize=9, title_fontsize=10)
        fig.tight_layout()

        path = os.path.join(output_dir, f"param_{param_name.lower()}_{func_name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")


# -----------------------------------------------
# Plot 2: final best_fitness boxplot per param_value
#   gives a clean summary without time axis
# -----------------------------------------------
def plot_param_boxplot(csv_path, output_dir, functions=FUNCTIONS):
    """
    Box plot: x = param_values, y = best_fitness distribution across trials.
    One subplot per function.
    """
    df         = pd.read_csv(csv_path)
    param_name = df["param_name"].iloc[0]
    param_values = list(df["param_value"].unique())

    os.makedirs(output_dir, exist_ok=True)

    n_funcs = len(functions)
    fig, axes = plt.subplots(1, n_funcs, figsize=(6 * n_funcs, 5))
    if n_funcs == 1:
        axes = [axes]

    fig.suptitle(f"Best Fitness Distribution: {param_name}", fontsize=14, fontweight="bold")

    for ax, func_name in zip(axes, functions):
        sub = df[df["func_name"] == func_name]
        if sub.empty:
            ax.set_title(f"{func_name} (no data)")
            continue

        data_per_value = [
            sub[sub["param_value"] == pv]["best_fitness"].values
            for pv in param_values
        ]

        bp = ax.boxplot(data_per_value, patch_artist=True, notch=False)
        colors = get_colors(len(param_values))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(range(1, len(param_values) + 1))
        ax.set_xticklabels([str(v) for v in param_values], rotation=25, ha="right", fontsize=8)
        ax.set_title(func_name.capitalize(), fontsize=12)
        ax.set_xlabel(param_name, fontsize=10)
        ax.set_ylabel("Best Fitness" if func_name == functions[0] else "", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = os.path.join(output_dir, f"param_{param_name.lower()}_boxplot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# -----------------------------------------------
# Run all CSV files in a folder
# -----------------------------------------------
def plot_all_param_files(input_folder, output_dir, functions=FUNCTIONS):
    """
    Scans input_folder for all *_param_*.csv files and plots each one.

    Parameters
    ----------
    input_folder : folder containing abc_param_*.csv, pso_param_*.csv, etc.
    output_dir   : folder to save all plots
    functions    : list of objective function names
    """
    csv_files = [
        f for f in os.listdir(input_folder)
        if f.endswith(".csv") and "param_" in f
    ]

    if not csv_files:
        print(f"No param CSV files found in {input_folder}")
        return

    for file_name in sorted(csv_files):
        csv_path  = os.path.join(input_folder, file_name)
        algo_name = file_name.split("_param_")[0].upper()

        print(f"\n{'='*55}")
        print(f"  Plotting: {file_name}  [{algo_name}]")
        print(f"{'='*55}")

        sub_output = os.path.join(output_dir, algo_name.lower())
        plot_param_file(csv_path,    sub_output, functions)
        plot_param_boxplot(csv_path, sub_output, functions)

    print(f"\nAll parameter plots saved to ./{output_dir}/")


# -----------------------------------------------
# Example usage
# -----------------------------------------------
if __name__ == "__main__":

    # Option A: plot all files in a folder automatically
    plot_all_param_files(
        input_folder = "param_analysis_data/cuckoo_search",
        output_dir   = "plots_param_analysis",
        functions    = FUNCTIONS,
    )

    # Option B: plot a single CSV file manually
    # plot_param_file(
    #     csv_path   = "param_analysis_data/abc/abc_param_colony_size.csv",
    #     output_dir = "plots_param_analysis/abc",
    # )
    # plot_param_boxplot(
    #     csv_path   = "param_analysis_data/abc/abc_param_colony_size.csv",
    #     output_dir = "plots_param_analysis/abc",
    # )