"""
Convergence Comparison — All Algorithms (TSP + Knapsack + GraphColoring)
=========================================================================
TSP:           uses hardcoded best configs per algorithm (most sensitive param).
Knapsack:      auto-selects best config per algorithm (highest median best_cost).
GraphColoring: auto-selects best config per algorithm (lowest median best_cost).

Outputs
-------
  src/plots/Discreate/TSP/
    convergence_comparison_raw.pdf
    convergence_comparison_normalized.pdf

  src/plots/Discreate/Knapsack/
    kp_comparison_raw.pdf
    kp_comparison_normalized.pdf

  src/plots/Discreate/GraphColoring/
    gc_comparison_raw.pdf
    gc_comparison_normalized.pdf

Usage
-----
    .venv/bin/python3 plot_convergence_comparison.py             # all three
    .venv/bin/python3 plot_convergence_comparison.py --problem tsp
    .venv/bin/python3 plot_convergence_comparison.py --problem knapsack
    .venv/bin/python3 plot_convergence_comparison.py --problem graphcoloring
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Project imports ───────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from core.eval_norm_tsp import work_factor as tsp_work_factor
from core.eval_norm_knapsack import work_factor_kp
from core.eval_norm_gc import work_factor_gc

# ── Output directories ────────────────────────────────────────────────────────
TSP_SENS_DIR = os.path.join(ROOT, "src", "parameter sensitivity", "TSP")
TSP_OUT_DIR  = os.path.join(ROOT, "src", "plots", "Discreate", "TSP")
KP_SENS_DIR  = os.path.join(ROOT, "src", "parameter sensitivity", "Knapsack")
KP_OUT_DIR   = os.path.join(ROOT, "src", "plots", "Discreate", "Knapsack")
GC_SENS_DIR  = os.path.join(ROOT, "src", "parameter sensitivity", "GraphColoring")
GC_OUT_DIR   = os.path.join(ROOT, "src", "plots", "Discreate", "GraphColoring")
for d in (TSP_OUT_DIR, KP_OUT_DIR, GC_OUT_DIR):
    os.makedirs(d, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize":  (10, 6),
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "font.size":       12,
    "lines.linewidth": 2.0,
})

N_GRID = 500

# ── TSP: hardcoded best configs ───────────────────────────────────────────────
TSP_COLORS = {
    "SA":   "#1f77b4",
    "GA":   "#ff7f0e",
    "HC":   "#2ca02c",
    "TLBO": "#d62728",
    "ACO":  "#9467bd",
}

TSP_ALGORITHMS = {
    "SA": {
        "trace": f"{TSP_SENS_DIR}/SA/T0/SA_T0_1_trace.csv",
        "run":   f"{TSP_SENS_DIR}/SA/T0/SA_T0_1_run.csv",
        "label": "SA (T\u2080=10)",
        "sensitive_param": "T\u2080",
    },
    "GA": {
        "trace": f"{TSP_SENS_DIR}/GA/mutation_rate/GA_muta_3_trace.csv",
        "run":   f"{TSP_SENS_DIR}/GA/mutation_rate/GA_muta_3_run.csv",
        "label": "GA (mut=0.10)",
        "sensitive_param": "mutation_rate",
    },
    "HC": {
        "trace": f"{TSP_SENS_DIR}/HC/best/HC_best_trace.csv",
        "run":   f"{TSP_SENS_DIR}/HC/best/HC_best_run.csv",
        "label": "HC (best-improvement)",
        "sensitive_param": "mode",
    },
    "TLBO": {
        "trace": f"{TSP_SENS_DIR}/TLBO/pop_size/TLBO_pop_3_trace.csv",
        "run":   f"{TSP_SENS_DIR}/TLBO/pop_size/TLBO_pop_3_run.csv",
        "label": "TLBO (pop=120)",
        "sensitive_param": "pop_size",
    },
    "ACO": {
        "trace": f"{TSP_SENS_DIR}/ACO/base/ACO_base_trace.csv",
        "run":   f"{TSP_SENS_DIR}/ACO/base/ACO_base_run.csv",
        "label": "ACO (n_ants=30)",
        "sensitive_param": "\u03b1 (pheromone)",
    },
}

# ── Knapsack: algo metadata ───────────────────────────────────────────────────
KP_COLORS = {
    "SA_KP":   "#1f77b4",
    "GA_KP":   "#ff7f0e",
    "HC_KP":   "#2ca02c",
    "TLBO_KP": "#d62728",
    "ABC_KP":  "#9467bd",
}

KP_DISPLAY = {
    "SA_KP":   "SA",
    "GA_KP":   "GA",
    "HC_KP":   "HC",
    "TLBO_KP": "TLBO",
    "ABC_KP":  "ABC",
}

KP_SUBDIR = {
    "SA_KP":   "SA",
    "GA_KP":   "GA",
    "HC_KP":   "HC",
    "TLBO_KP": "TLBO",
    "ABC_KP":  "ABC",
}

# ── GraphColoring: algo metadata ──────────────────────────────────────────────
GC_COLORS = {
    "SA_GC":  "#1f77b4",
    "GA_GC":  "#ff7f0e",
    "HC_GC":  "#2ca02c",
    "ACO_GC": "#d62728",
    "DFS_GC": "#9467bd",
}

GC_DISPLAY = {
    "SA_GC":  "SA",
    "GA_GC":  "GA",
    "HC_GC":  "HC",
    "ACO_GC": "ACO",
    "DFS_GC": "DFS",
}

GC_SUBDIR = {
    "SA_GC":  "SA",
    "GA_GC":  "GA",
    "HC_GC":  "HC",
    "ACO_GC": "ACO",
    "DFS_GC": "DFS",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════════

def compute_interpolated_stats(df: pd.DataFrame, x_col: str):
    """Per-run interpolation onto a common grid → (grid, median, Q25, Q75)."""
    curves = []
    for _, run_df in df.groupby("run_id"):
        run_df = run_df.sort_values(x_col)
        mask = np.isfinite(run_df["best_cost"].values)
        xv = run_df[x_col].values[mask]
        yv = run_df["best_cost"].values[mask]
        if len(xv) >= 2:
            curves.append((xv, yv))
    if not curves:
        return None
    grid_min = max(c[0][0]  for c in curves)
    grid_max = min(c[0][-1] for c in curves)
    if grid_max <= grid_min:
        grid_max = max(c[0][-1] for c in curves)
        grid_min = min(c[0][0]  for c in curves)
    grid   = np.linspace(grid_min, grid_max, N_GRID)
    interp = np.empty((len(curves), N_GRID))
    for i, (xv, yv) in enumerate(curves):
        interp[i] = np.interp(grid, xv, yv)
    return (
        grid,
        np.median(interp,         axis=0),
        np.quantile(interp, 0.25, axis=0),
        np.quantile(interp, 0.75, axis=0),
    )


def plot_comparison(all_stats, labels, colors, title, xlabel, ylabel, out_path, algo_order):
    """Draw all algorithms on one convergence PDF."""
    fig, ax = plt.subplots()
    for algo in algo_order:
        if algo not in all_stats or all_stats[algo] is None:
            continue
        x, med, q25, q75 = all_stats[algo]
        color = colors[algo]
        ax.plot(x, med, color=color, label=labels[algo])
        ax.fill_between(x, q25, q75, color=color, alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# TSP
# ═══════════════════════════════════════════════════════════════════════════════

def load_tsp_data(algo_name: str, info: dict) -> pd.DataFrame:
    trace_df = pd.read_csv(info["trace"])
    run_df   = pd.read_csv(info["run"])
    if "n_cities" not in trace_df.columns:
        mapping = run_df.set_index("run_id")["n_cities"].to_dict()
        trace_df["n_cities"] = trace_df["run_id"].map(mapping)
    n  = int(trace_df["n_cities"].iloc[0]) if "n_cities" in trace_df.columns else 30
    wf = tsp_work_factor(algo_name, n)
    trace_df["norm_evals"] = trace_df["evals_cost"].astype(float) * wf
    return trace_df


def run_tsp():
    print("── TSP (n=30) ─────────────────────────────────────────────────────")
    print("Selected configs (most sensitive param, best value):")
    for algo, info in TSP_ALGORITHMS.items():
        print(f"  {algo:5s} — sensitive: {info['sensitive_param']:15s} → {info['label']}")
    print()

    algo_order = ["HC", "SA", "GA", "TLBO", "ACO"]
    raw_stats  = {}
    norm_stats = {}
    labels     = {a: TSP_ALGORITHMS[a]["label"] for a in algo_order}

    for algo in algo_order:
        info = TSP_ALGORITHMS[algo]
        if not os.path.exists(info["trace"]):
            print(f"  SKIP {algo}: trace file not found")
            continue
        df = load_tsp_data(algo, info)
        raw_stats[algo]  = compute_interpolated_stats(df, "evals_cost")
        norm_stats[algo] = compute_interpolated_stats(df, "norm_evals")
        r = raw_stats[algo]
        if r:
            print(f"  {algo}: raw evals [{r[0][0]:.0f}, {r[0][-1]:.0f}], "
                  f"final median = {r[1][-1]:.2f}")

    print()
    plot_comparison(
        raw_stats, labels, TSP_COLORS,
        title="Convergence Comparison — All Algorithms (TSP n=30, raw evals)",
        xlabel="Evaluations (raw evals_cost)",
        ylabel="Best Cost",
        out_path=os.path.join(TSP_OUT_DIR, "convergence_comparison_raw.pdf"),
        algo_order=algo_order,
    )
    plot_comparison(
        norm_stats, labels, TSP_COLORS,
        title="Convergence Comparison — All Algorithms (TSP n=30, normalized)",
        xlabel="Normalized Evaluations (O(n)-equivalent)",
        ylabel="Best Cost",
        out_path=os.path.join(TSP_OUT_DIR, "convergence_comparison_normalized.pdf"),
        algo_order=algo_order,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Knapsack
# ═══════════════════════════════════════════════════════════════════════════════

def find_best_kp_param(algo: str) -> dict | None:
    """Return info dict for the run CSV with the highest median best_cost."""
    algo_dir = os.path.join(KP_SENS_DIR, KP_SUBDIR[algo])
    run_csvs = sorted(glob.glob(os.path.join(algo_dir, "**", "*_run.csv"), recursive=True))
    if not run_csvs:
        return None
    best_run_csv = None
    best_score   = -np.inf
    best_label   = ""
    for run_csv in run_csvs:
        try:
            df = pd.read_csv(run_csv)
        except Exception:
            continue
        if "best_cost" not in df.columns or df.empty:
            continue
        score = float(df["best_cost"].median())
        if score > best_score:
            best_score   = score
            best_run_csv = run_csv
            tag = (str(df["param_tag"].iloc[0]) if "param_tag" in df.columns
                   else os.path.basename(run_csv).replace("_run.csv", ""))
            best_label = f"{KP_DISPLAY[algo]} ({tag})"
    if best_run_csv is None:
        return None
    trace_csv = best_run_csv.replace("_run.csv", "_trace.csv")
    if not os.path.exists(trace_csv):
        return None
    return {
        "trace": trace_csv,
        "run":   best_run_csv,
        "label": best_label,
        "tag":   os.path.basename(best_run_csv).replace("_run.csv", ""),
        "score": best_score,
    }


def load_kp_data(algo: str, info: dict) -> pd.DataFrame:
    trace_df = pd.read_csv(info["trace"])
    trace_df = trace_df.drop_duplicates(subset=["run_id", "iter"], keep="last")
    if "normalized_evals" not in trace_df.columns or trace_df["normalized_evals"].isna().all():
        trace_df["normalized_evals"] = (
            trace_df["evals_cost"].astype(float) * work_factor_kp(algo)
        )
    return trace_df


def run_knapsack():
    print("── Knapsack (n=100) ──────────────────────────────────────────────────")
    print("Auto-selected best configurations (highest median best_cost):")

    algo_order = ["HC_KP", "SA_KP", "GA_KP", "TLBO_KP", "ABC_KP"]
    algo_info  = {}
    raw_stats  = {}
    norm_stats = {}
    labels     = {}

    for algo in algo_order:
        info = find_best_kp_param(algo)
        if info is None:
            print(f"  SKIP {algo}: no CSV data found")
            continue
        algo_info[algo] = info
        labels[algo]    = info["label"]
        print(f"  {KP_DISPLAY[algo]:5s} — best: {info['tag']:<40s} median={info['score']:.1f}")

    print()
    for algo in algo_order:
        if algo not in algo_info:
            continue
        df = load_kp_data(algo, algo_info[algo])
        raw_stats[algo]  = compute_interpolated_stats(df, "evals_cost")
        norm_stats[algo] = compute_interpolated_stats(df, "normalized_evals")
        r = raw_stats[algo]
        if r is not None:
            print(f"  {algo}: raw evals [{r[0][0]:.0f}, {r[0][-1]:.0f}], "
                  f"final median = {r[1][-1]:.2f}")

    print()
    plot_comparison(
        raw_stats, labels, KP_COLORS,
        title="Convergence Comparison — All Algorithms (Knapsack n=100, raw evals)",
        xlabel="Evaluations (raw evals_cost)",
        ylabel="Best Cost (higher is better)",
        out_path=os.path.join(KP_OUT_DIR, "kp_comparison_raw.pdf"),
        algo_order=algo_order,
    )
    plot_comparison(
        norm_stats, labels, KP_COLORS,
        title="Convergence Comparison — All Algorithms (Knapsack n=100, normalized)",
        xlabel="Normalized Evaluations (O(n)-equivalent)",
        ylabel="Best Cost (higher is better)",
        out_path=os.path.join(KP_OUT_DIR, "kp_comparison_normalized.pdf"),
        algo_order=algo_order,
    )


# ═══════════════════════════════════════════════════════════════════
# GraphColoring
# ═══════════════════════════════════════════════════════════════════

def _infer_n_gc(run_id: str) -> int:
    import re
    m = re.search(r"_n(\d+)_", str(run_id))
    return int(m.group(1)) if m else 30


def find_best_gc_param(algo: str) -> dict | None:
    """Return info dict for the run CSV with the lowest median best_cost (minimization)."""
    algo_dir = os.path.join(GC_SENS_DIR, GC_SUBDIR[algo])
    run_csvs = sorted(glob.glob(os.path.join(algo_dir, "**", "*_run.csv"), recursive=True))
    if not run_csvs:
        return None
    best_run_csv = None
    best_score   = np.inf
    for run_csv in run_csvs:
        try:
            df = pd.read_csv(run_csv)
        except Exception:
            continue
        if "best_cost" not in df.columns or df.empty:
            continue
        score = float(df["best_cost"].median())
        if score < best_score:   # lower = fewer conflicts = better
            best_score   = score
            best_run_csv = run_csv
            tag = (str(df["param_tag"].iloc[0]) if "param_tag" in df.columns
                   else os.path.basename(run_csv).replace("_run.csv", ""))
            best_label = f"{GC_DISPLAY[algo]} ({tag})"
    if best_run_csv is None:
        return None
    trace_csv = best_run_csv.replace("_run.csv", "_trace.csv")
    if not os.path.exists(trace_csv):
        return None
    return {
        "trace": trace_csv,
        "run":   best_run_csv,
        "label": best_label,
        "tag":   os.path.basename(best_run_csv).replace("_run.csv", ""),
        "score": best_score,
    }


def load_gc_data(algo: str, info: dict) -> pd.DataFrame:
    trace_df = pd.read_csv(info["trace"])
    trace_df = trace_df.drop_duplicates(subset=["run_id", "iter"], keep="last")
    if "normalized_evals" not in trace_df.columns or trace_df["normalized_evals"].isna().all():
        n = _infer_n_gc(trace_df["run_id"].iloc[0]) if "run_id" in trace_df.columns else 30
        trace_df["normalized_evals"] = (
            trace_df["evals_cost"].astype(float) * work_factor_gc(algo, n)
        )
    return trace_df


def run_graphcoloring():
    print("── GraphColoring (n=30) ─────────────────────────────────────────────")
    print("Auto-selected best configurations (lowest median best_cost = fewest conflicts):")

    algo_order = ["HC_GC", "SA_GC", "GA_GC", "ACO_GC", "DFS_GC"]
    algo_info  = {}
    raw_stats  = {}
    norm_stats = {}
    labels     = {}

    for algo in algo_order:
        info = find_best_gc_param(algo)
        if info is None:
            print(f"  SKIP {algo}: no CSV data found")
            continue
        algo_info[algo] = info
        labels[algo]    = info["label"]
        print(f"  {GC_DISPLAY[algo]:5s} — best: {info['tag']:<40s} median={info['score']:.2f}")

    print()
    for algo in algo_order:
        if algo not in algo_info:
            continue
        df = load_gc_data(algo, algo_info[algo])
        raw_stats[algo]  = compute_interpolated_stats(df, "evals_cost")
        norm_stats[algo] = compute_interpolated_stats(df, "normalized_evals")
        r = raw_stats[algo]
        if r is not None:
            print(f"  {algo}: raw evals [{r[0][0]:.0f}, {r[0][-1]:.0f}], "
                  f"final median = {r[1][-1]:.2f}")

    print()
    plot_comparison(
        raw_stats, labels, GC_COLORS,
        title="Convergence Comparison — All Algorithms (GraphColoring n=30, raw evals)",
        xlabel="Evaluations (raw evals_cost)",
        ylabel="Best Cost (lower is better)",
        out_path=os.path.join(GC_OUT_DIR, "gc_comparison_raw.pdf"),
        algo_order=algo_order,
    )
    plot_comparison(
        norm_stats, labels, GC_COLORS,
        title="Convergence Comparison — All Algorithms (GraphColoring n=30, normalized)",
        xlabel="Normalized Evaluations (O(n)-equivalent)",
        ylabel="Best Cost (lower is better)",
        out_path=os.path.join(GC_OUT_DIR, "gc_comparison_normalized.pdf"),
        algo_order=algo_order,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convergence comparison plots for TSP, Knapsack, and/or GraphColoring"
    )
    parser.add_argument(
        "--problem", nargs="*", default=None,
        help="Problem(s) to plot: tsp knapsack graphcoloring  (default: all)",
    )
    args = parser.parse_args()

    selected = {p.lower() for p in args.problem} if args.problem else {"tsp", "knapsack", "graphcoloring"}

    if "tsp" in selected:
        run_tsp()
        print()
    if "knapsack" in selected:
        run_knapsack()
        print()
    if "graphcoloring" in selected:
        run_graphcoloring()
        print()

    print("Done!")


if __name__ == "__main__":
    main()
