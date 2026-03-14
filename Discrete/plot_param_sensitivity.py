"""Parameter-sensitivity convergence plots for TSP, Knapsack, and GraphColoring.

Usage:
    python plot_param_sensitivity.py --problem TSP
    python plot_param_sensitivity.py --problem Knapsack
    python plot_param_sensitivity.py --problem GraphColoring
    python plot_param_sensitivity.py --problem TSP --exp SA_T0 GA_pop
    python plot_param_sensitivity.py --problem GraphColoring --exp SA_T0 ACO_n_ants
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


# ── Directories ───────────────────────────────────────────────────────────────

def _sens_dir(problem: str) -> str:
    return os.path.join(ROOT, "src", "parameter sensitivity", problem)


def _out_dir(problem: str) -> str:
    d = os.path.join(ROOT, "src", "plots", "Discreate", problem)
    os.makedirs(d, exist_ok=True)
    return d


# ── Normalization helpers ─────────────────────────────────────────────────────

def _add_norm_evals_tsp(df: pd.DataFrame, algo: str) -> pd.DataFrame:
    from core.eval_norm_tsp import work_factor
    out = df.copy()
    if "normalized_evals" in out.columns:
        out["norm_evals"] = out["normalized_evals"].astype(float)
    else:
        # TSP work_factor needs (algo, n) — default n=30
        n = int(out["n_cities"].iloc[0]) if "n_cities" in out.columns else 30
        factor = work_factor(algo, n)
        out["norm_evals"] = out["evals_cost"].astype(float) * factor
    return out


def _add_norm_evals_knapsack(df: pd.DataFrame, algo: str) -> pd.DataFrame:
    from core.eval_norm_knapsack import work_factor_kp
    out = df.copy()
    if "normalized_evals" in out.columns:
        out["norm_evals"] = out["normalized_evals"].astype(float)
    else:
        factor = work_factor_kp(algo)
        out["norm_evals"] = out["evals_cost"].astype(float) * factor
    return out


def _add_norm_evals_gc(df: pd.DataFrame, algo: str) -> pd.DataFrame:
    import re
    from core.eval_norm_gc import work_factor_gc
    out = df.copy()
    if "normalized_evals" in out.columns and not out["normalized_evals"].isna().all():
        out["norm_evals"] = out["normalized_evals"].astype(float)
    else:
        # Extract n_nodes from run_id (format: algo[tag]_n30_inst...) or default 30
        def _infer_n(run_id: str) -> int:
            m = re.search(r"_n(\d+)_", str(run_id))
            return int(m.group(1)) if m else 30
        n = int(out["run_id"].iloc[0] if "run_id" in out.columns else 30)
        try:
            n = _infer_n(out["run_id"].iloc[0])
        except Exception:
            n = 30
        factor = work_factor_gc(algo, n)
        out["norm_evals"] = out["evals_cost"].astype(float) * factor
    return out


# ── Experiment definitions ────────────────────────────────────────────────────

def _tsp_experiments(d: str) -> dict:
    return {
        "SA_T0": {
            "algo": "SA",
            "param": "T0",
            "files": [
                (f"{d}/SA/T0/SA_T0_1_trace.csv", "T0 = 10"),
                (f"{d}/SA/base/SA_base_trace.csv", "T0 = 100 (default)"),
                (f"{d}/SA/T0/SA_T0_2_trace.csv", "T0 = 500"),
                (f"{d}/SA/T0/SA_T0_3_trace.csv", "T0 = 1000"),
            ],
            "prefix": "tsp_sa_convergence_T0",
        },
        "SA_steps_per_temp": {
            "algo": "SA",
            "param": "steps_per_temp",
            "files": [
                (f"{d}/SA/steps_per_temp/SA_step_1_trace.csv", "steps = 500"),
                (f"{d}/SA/steps_per_temp/SA_step_2_trace.csv", "steps = 1000"),
                (f"{d}/SA/base/SA_base_trace.csv", "steps = 2000 (default)"),
                (f"{d}/SA/steps_per_temp/SA_step_3_trace.csv", "steps = 4000"),
            ],
            "prefix": "tsp_sa_convergence_steps_per_temp",
        },
        "GA_pop": {
            "algo": "GA",
            "param": "population_size",
            "files": [
                (f"{d}/GA/pop/GA_pop_1_trace.csv", "pop = 20"),
                (f"{d}/GA/pop/GA_pop_2_trace.csv", "pop = 50"),
                (f"{d}/GA/base/GA_base_trace.csv", "pop = 100 (default)"),
                (f"{d}/GA/pop/GA_pop_3_trace.csv", "pop = 500"),
            ],
            "prefix": "tsp_ga_convergence_pop",
        },
        "GA_crossover": {
            "algo": "GA",
            "param": "crossover_rate",
            "files": [
                (f"{d}/GA/crossover/GA_cross_1_trace.csv", "cx = 0.6"),
                (f"{d}/GA/crossover/GA_cross_2_trace.csv", "cx = 0.7"),
                (f"{d}/GA/crossover/GA_cross_base_trace.csv", "cx = 0.8 (default)"),
                (f"{d}/GA/crossover/GA_cross_3_trace.csv", "cx = 0.9"),
            ],
            "prefix": "tsp_ga_convergence_crossover",
        },
        "GA_mutation": {
            "algo": "GA",
            "param": "mutation_rate",
            "files": [
                (f"{d}/GA/mutation_rate/GA_muta_1_trace.csv", "mut = 0.01"),
                (f"{d}/GA/base/GA_base_trace.csv", "mut = 0.02 (default)"),
                (f"{d}/GA/mutation_rate/GA_muta_2_trace.csv", "mut = 0.05"),
                (f"{d}/GA/mutation_rate/GA_muta_3_trace.csv", "mut = 0.10"),
            ],
            "prefix": "tsp_ga_convergence_mutation",
        },
        "GA_tournament_k": {
            "algo": "GA",
            "param": "tournament_k",
            "files": [
                (f"{d}/GA/tournament_k/GA_tour_1_trace.csv", "k = 3"),
                (f"{d}/GA/base/GA_base_trace.csv", "k = 5 (default)"),
                (f"{d}/GA/tournament_k/GA_tour_2_trace.csv", "k = 10"),
                (f"{d}/GA/tournament_k/GA_tour_3_trace.csv", "k = 20"),
            ],
            "prefix": "tsp_ga_convergence_tournament_k",
        },
        "HC_mode": {
            "algo": "HC",
            "param": "mode",
            "files": [
                (f"{d}/HC/best/HC_best_trace.csv", "best-improvement"),
                (f"{d}/HC/first/HC_first_trace.csv", "first-improvement"),
            ],
            "prefix": "tsp_hc_convergence_mode",
        },
        "TLBO_pop": {
            "algo": "TLBO",
            "param": "pop_size",
            "files": [
                (f"{d}/TLBO/pop_size/TLBO_pop_1_trace.csv", "pop = 30"),
                (f"{d}/TLBO/base/TLBO_base_trace.csv", "pop = 50 (default)"),
                (f"{d}/TLBO/pop_size/TLBO_pop_2_trace.csv", "pop = 80"),
                (f"{d}/TLBO/pop_size/TLBO_pop_3_trace.csv", "pop = 120"),
            ],
            "prefix": "tsp_tlbo_convergence_pop",
        },
        "TLBO_move_frac_max": {
            "algo": "TLBO",
            "param": "move_frac_max",
            "files": [
                (f"{d}/TLBO/move_frac_max/TLBO_move_1_trace.csv", "frac = 0.2"),
                (f"{d}/TLBO/base/TLBO_base_trace.csv", "frac = 0.4 (default)"),
                (f"{d}/TLBO/move_frac_max/TLBO_move_2_trace.csv", "frac = 0.6"),
                (f"{d}/TLBO/move_frac_max/TLBO_move_3_trace.csv", "frac = 0.8"),
            ],
            "prefix": "tsp_tlbo_convergence_move_frac",
        },
        "ACO_n_ants": {
            "algo": "ACO",
            "param": "n_ants",
            "files": [
                (f"{d}/ACO/n_ants/ACO_nants_1_trace.csv", "ants = 10"),
                (f"{d}/ACO/n_ants/ACO_nants_2_trace.csv", "ants = 20"),
                (f"{d}/ACO/base/ACO_base_trace.csv", "ants = 30 (default)"),
                (f"{d}/ACO/n_ants/ACO_nants_3_trace.csv", "ants = 50"),
            ],
            "prefix": "tsp_aco_convergence_n_ants",
        },
        "ACO_alpha": {
            "algo": "ACO",
            "param": "alpha",
            "files": [
                (f"{d}/ACO/alpha/ACO_alpha_1_trace.csv", "α = 0.5"),
                (f"{d}/ACO/base/ACO_base_trace.csv", "α = 1.0 (default)"),
                (f"{d}/ACO/alpha/ACO_alpha_2_trace.csv", "α = 2.0"),
                (f"{d}/ACO/alpha/ACO_alpha_3_trace.csv", "α = 3.0"),
            ],
            "prefix": "tsp_aco_convergence_alpha",
        },
        "ACO_beta": {
            "algo": "ACO",
            "param": "beta",
            "files": [
                (f"{d}/ACO/beta/ACO_beta_1_trace.csv", "β = 1"),
                (f"{d}/ACO/beta/ACO_beta_2_trace.csv", "β = 2"),
                (f"{d}/ACO/base/ACO_base_trace.csv", "β = 3 (default)"),
                (f"{d}/ACO/beta/ACO_beta_3_trace.csv", "β = 5"),
            ],
            "prefix": "tsp_aco_convergence_beta",
        },
        "ACO_rho": {
            "algo": "ACO",
            "param": "rho",
            "files": [
                (f"{d}/ACO/rho/ACO_rho_1_trace.csv", "ρ = 0.05"),
                (f"{d}/ACO/base/ACO_base_trace.csv", "ρ = 0.1 (default)"),
                (f"{d}/ACO/rho/ACO_rho_2_trace.csv", "ρ = 0.2"),
                (f"{d}/ACO/rho/ACO_rho_3_trace.csv", "ρ = 0.5"),
            ],
            "prefix": "tsp_aco_convergence_rho",
        },
    }


def _gc_experiments(d: str) -> dict:
    return {
        # ── SA_GC ──────────────────────────────────────────────────────────
        "SA_T0": {
            "algo": "SA_GC",
            "param": "T0",
            "files": [
                (f"{d}/SA/T0/GC_SA_T0_1_trace.csv",    "T0 = 1.0"),
                (f"{d}/SA/T0/GC_SA_T0_2_trace.csv",    "T0 = 10.0"),
                (f"{d}/SA/base/GC_SA_base_trace.csv",   "T0 = 5.0 (default)"),
                (f"{d}/SA/T0/GC_SA_T0_3_trace.csv",    "T0 = 50.0"),
            ],
            "prefix": "gc_sa_convergence_T0",
        },
        "SA_alpha": {
            "algo": "SA_GC",
            "param": "alpha",
            "files": [
                (f"{d}/SA/alpha/GC_SA_alpha_1_trace.csv", "α = 0.95"),
                (f"{d}/SA/alpha/GC_SA_alpha_2_trace.csv", "α = 0.97"),
                (f"{d}/SA/base/GC_SA_base_trace.csv",     "α = 0.99 (default)"),
                (f"{d}/SA/alpha/GC_SA_alpha_3_trace.csv", "α = 0.999"),
            ],
            "prefix": "gc_sa_convergence_alpha",
        },
        # ── GA_GC ──────────────────────────────────────────────────────────
        "GA_pop": {
            "algo": "GA_GC",
            "param": "population_size",
            "files": [
                (f"{d}/GA/pop/GC_GA_pop_1_trace.csv",  "pop = 20"),
                (f"{d}/GA/pop/GC_GA_pop_2_trace.csv",  "pop = 50"),
                (f"{d}/GA/base/GC_GA_base_trace.csv",   "pop = 100 (default)"),
                (f"{d}/GA/pop/GC_GA_pop_3_trace.csv",  "pop = 200"),
            ],
            "prefix": "gc_ga_convergence_pop",
        },
        "GA_crossover": {
            "algo": "GA_GC",
            "param": "crossover_rate",
            "files": [
                (f"{d}/GA/crossover/GC_GA_cross_1_trace.csv", "cx = 0.6"),
                (f"{d}/GA/crossover/GC_GA_cross_2_trace.csv", "cx = 0.7"),
                (f"{d}/GA/base/GC_GA_base_trace.csv",          "cx = 0.9 (default)"),
                (f"{d}/GA/crossover/GC_GA_cross_3_trace.csv", "cx = 0.95"),
            ],
            "prefix": "gc_ga_convergence_crossover",
        },
        "GA_mutation": {
            "algo": "GA_GC",
            "param": "mutation_rate",
            "files": [
                (f"{d}/GA/mutation_rate/GC_GA_muta_1_trace.csv", "mut = 0.01"),
                (f"{d}/GA/base/GC_GA_base_trace.csv",             "mut = 0.02 (default)"),
                (f"{d}/GA/mutation_rate/GC_GA_muta_2_trace.csv", "mut = 0.05"),
                (f"{d}/GA/mutation_rate/GC_GA_muta_3_trace.csv", "mut = 0.10"),
            ],
            "prefix": "gc_ga_convergence_mutation",
        },
        "GA_tournament_k": {
            "algo": "GA_GC",
            "param": "tournament_k",
            "files": [
                (f"{d}/GA/tournament_k/GC_GA_tour_1_trace.csv", "k = 3"),
                (f"{d}/GA/tournament_k/GC_GA_tour_2_trace.csv", "k = 5"),
                (f"{d}/GA/base/GC_GA_base_trace.csv",            "k = 10 (default)"),
                (f"{d}/GA/tournament_k/GC_GA_tour_3_trace.csv", "k = 20"),
            ],
            "prefix": "gc_ga_convergence_tournament_k",
        },
        # ── HC_GC ──────────────────────────────────────────────────────────
        "HC_mode": {
            "algo": "HC_GC",
            "param": "mode",
            "files": [
                (f"{d}/HC/best/GC_HC_best_trace.csv",   "best-improvement"),
                (f"{d}/HC/first/GC_HC_first_trace.csv", "first-improvement"),
            ],
            "prefix": "gc_hc_convergence_mode",
        },
        # ── ACO_GC ─────────────────────────────────────────────────────────
        "ACO_n_ants": {
            "algo": "ACO_GC",
            "param": "n_ants",
            "files": [
                (f"{d}/ACO/n_ants/GC_ACO_nants_1_trace.csv", "ants = 10"),
                (f"{d}/ACO/n_ants/GC_ACO_nants_2_trace.csv", "ants = 20"),
                (f"{d}/ACO/base/GC_ACO_base_trace.csv",       "ants = 30 (default)"),
                (f"{d}/ACO/n_ants/GC_ACO_nants_3_trace.csv", "ants = 50"),
            ],
            "prefix": "gc_aco_convergence_n_ants",
        },
        "ACO_alpha": {
            "algo": "ACO_GC",
            "param": "alpha",
            "files": [
                (f"{d}/ACO/alpha/GC_ACO_alpha_1_trace.csv", "α = 0.5"),
                (f"{d}/ACO/base/GC_ACO_base_trace.csv",      "α = 1.0 (default)"),
                (f"{d}/ACO/alpha/GC_ACO_alpha_2_trace.csv", "α = 2.0"),
                (f"{d}/ACO/alpha/GC_ACO_alpha_3_trace.csv", "α = 3.0"),
            ],
            "prefix": "gc_aco_convergence_alpha",
        },
        "ACO_beta": {
            "algo": "ACO_GC",
            "param": "beta",
            "files": [
                (f"{d}/ACO/beta/GC_ACO_beta_1_trace.csv", "β = 1.0"),
                (f"{d}/ACO/base/GC_ACO_base_trace.csv",   "β = 2.0 (default)"),
                (f"{d}/ACO/beta/GC_ACO_beta_2_trace.csv", "β = 3.0"),
                (f"{d}/ACO/beta/GC_ACO_beta_3_trace.csv", "β = 5.0"),
            ],
            "prefix": "gc_aco_convergence_beta",
        },
        "ACO_rho": {
            "algo": "ACO_GC",
            "param": "rho",
            "files": [
                (f"{d}/ACO/rho/GC_ACO_rho_1_trace.csv", "ρ = 0.05"),
                (f"{d}/ACO/base/GC_ACO_base_trace.csv",  "ρ = 0.1 (default)"),
                (f"{d}/ACO/rho/GC_ACO_rho_2_trace.csv", "ρ = 0.2"),
                (f"{d}/ACO/rho/GC_ACO_rho_3_trace.csv", "ρ = 0.5"),
            ],
            "prefix": "gc_aco_convergence_rho",
        },
        # ── DFS_GC ───────────────────────────────────────────────────────
        "DFS_time_limit": {
            "algo": "DFS_GC",
            "param": "time_limit_sec",
            "files": [
                (f"{d}/DFS/time_limit/GC_DFS_time_1_trace.csv", "limit = 0.5s"),
                (f"{d}/DFS/time_limit/GC_DFS_time_2_trace.csv", "limit = 1.0s"),
                (f"{d}/DFS/base/GC_DFS_base_trace.csv",         "limit = 2.0s (default)"),
                (f"{d}/DFS/time_limit/GC_DFS_time_3_trace.csv", "limit = 5.0s"),
            ],
            "prefix": "gc_dfs_convergence_time_limit",
        },
        "DFS_max_backtracks": {
            "algo": "DFS_GC",
            "param": "max_backtracks",
            "files": [
                (f"{d}/DFS/max_backtracks/GC_DFS_bt_1_trace.csv", "bt = 100k"),
                (f"{d}/DFS/max_backtracks/GC_DFS_bt_2_trace.csv", "bt = 500k"),
                (f"{d}/DFS/base/GC_DFS_base_trace.csv",           "bt = 2M (default)"),
                (f"{d}/DFS/max_backtracks/GC_DFS_bt_3_trace.csv", "bt = 5M"),
            ],
            "prefix": "gc_dfs_convergence_max_backtracks",
        },
    }


def _knapsack_experiments(d: str) -> dict:
    return {
        "SA_T0": {
            "algo": "SA_KP",
            "param": "T0",
            "files": [
                (f"{d}/SA/T0/knapsack_SA_T0_1_trace.csv", "T0 = 10"),
                (f"{d}/SA/base/knapsack_SA_base_trace.csv", "T0 = 100 (default)"),
                (f"{d}/SA/T0/knapsack_SA_T0_2_trace.csv", "T0 = 500"),
                (f"{d}/SA/T0/knapsack_SA_T0_3_trace.csv", "T0 = 1000"),
            ],
            "prefix": "knapsack_sa_convergence_T0",
        },
        "SA_steps_per_temp": {
            "algo": "SA_KP",
            "param": "steps_per_temp",
            "files": [
                (f"{d}/SA/steps_per_temp/knapsack_SA_step_1_trace.csv", "steps = 500"),
                (f"{d}/SA/steps_per_temp/knapsack_SA_step_2_trace.csv", "steps = 1000"),
                (f"{d}/SA/base/knapsack_SA_base_trace.csv", "steps = 2000 (default)"),
                (f"{d}/SA/steps_per_temp/knapsack_SA_step_3_trace.csv", "steps = 4000"),
            ],
            "prefix": "knapsack_sa_convergence_steps_per_temp",
        },
        "GA_pop": {
            "algo": "GA_KP",
            "param": "population_size",
            "files": [
                (f"{d}/GA/pop/knapsack_GA_pop_1_trace.csv", "pop = 20"),
                (f"{d}/GA/pop/knapsack_GA_pop_2_trace.csv", "pop = 50"),
                (f"{d}/GA/base/knapsack_GA_base_trace.csv", "pop = 100 (default)"),
                (f"{d}/GA/pop/knapsack_GA_pop_3_trace.csv", "pop = 500"),
            ],
            "prefix": "knapsack_ga_convergence_pop",
        },
        "GA_crossover": {
            "algo": "GA_KP",
            "param": "crossover_rate",
            "files": [
                (f"{d}/GA/crossover/knapsack_GA_cross_1_trace.csv", "cx = 0.6"),
                (f"{d}/GA/crossover/knapsack_GA_cross_2_trace.csv", "cx = 0.7"),
                (f"{d}/GA/base/knapsack_GA_base_trace.csv", "cx = 0.8 (default)"),
                (f"{d}/GA/crossover/knapsack_GA_cross_3_trace.csv", "cx = 0.9"),
            ],
            "prefix": "knapsack_ga_convergence_crossover",
        },
        "GA_mutation": {
            "algo": "GA_KP",
            "param": "mutation_rate",
            "files": [
                (f"{d}/GA/mutation_rate/knapsack_GA_mut_1_trace.csv", "mut = 0.01"),
                (f"{d}/GA/base/knapsack_GA_base_trace.csv", "mut = 0.02 (default)"),
                (f"{d}/GA/mutation_rate/knapsack_GA_mut_2_trace.csv", "mut = 0.05"),
                (f"{d}/GA/mutation_rate/knapsack_GA_mut_3_trace.csv", "mut = 0.10"),
            ],
            "prefix": "knapsack_ga_convergence_mutation",
        },
        "GA_tournament_k": {
            "algo": "GA_KP",
            "param": "tournament_k",
            "files": [
                (f"{d}/GA/tournament_k/knapsack_GA_tour_1_trace.csv", "k = 3"),
                (f"{d}/GA/base/knapsack_GA_base_trace.csv", "k = 5 (default)"),
                (f"{d}/GA/tournament_k/knapsack_GA_tour_2_trace.csv", "k = 10"),
                (f"{d}/GA/tournament_k/knapsack_GA_tour_3_trace.csv", "k = 20"),
            ],
            "prefix": "knapsack_ga_convergence_tournament_k",
        },
        "HC_mode": {
            "algo": "HC_KP",
            "param": "mode",
            "files": [
                (f"{d}/HC/best/knapsack_HC_best_trace.csv", "best-improvement"),
                (f"{d}/HC/first/knapsack_HC_first_trace.csv", "first-improvement"),
            ],
            "prefix": "knapsack_hc_convergence_mode",
        },
        "TLBO_pop": {
            "algo": "TLBO_KP",
            "param": "pop_size",
            "files": [
                (f"{d}/TLBO/pop_size/knapsack_TLBO_pop_1_trace.csv", "pop = 30"),
                (f"{d}/TLBO/base/knapsack_TLBO_base_trace.csv", "pop = 50 (default)"),
                (f"{d}/TLBO/pop_size/knapsack_TLBO_pop_2_trace.csv", "pop = 80"),
                (f"{d}/TLBO/pop_size/knapsack_TLBO_pop_3_trace.csv", "pop = 120"),
            ],
            "prefix": "knapsack_tlbo_convergence_pop",
        },
        "TLBO_move_frac_max": {
            "algo": "TLBO_KP",
            "param": "move_frac_max",
            "files": [
                (f"{d}/TLBO/move_frac_max/knapsack_TLBO_move_1_trace.csv", "frac = 0.2"),
                (f"{d}/TLBO/base/knapsack_TLBO_base_trace.csv", "frac = 0.4 (default)"),
                (f"{d}/TLBO/move_frac_max/knapsack_TLBO_move_2_trace.csv", "frac = 0.6"),
                (f"{d}/TLBO/move_frac_max/knapsack_TLBO_move_3_trace.csv", "frac = 0.8"),
            ],
            "prefix": "knapsack_tlbo_convergence_move_frac_max",
        },
        "ABC_sn": {
            "algo": "ABC_KP",
            "param": "sn",
            "files": [
                (f"{d}/ABC/sn/knapsack_ABC_sn_1_trace.csv", "sn = 10"),
                (f"{d}/ABC/base/knapsack_ABC_base_trace.csv", "sn = 50 (default)"),
                (f"{d}/ABC/sn/knapsack_ABC_sn_2_trace.csv", "sn = 100"),
                (f"{d}/ABC/sn/knapsack_ABC_sn_3_trace.csv", "sn = 500"),
            ],
            "prefix": "knapsack_abc_convergence_sn",
        },
        "ABC_limit": {
            "algo": "ABC_KP",
            "param": "limit",
            "files": [
                (f"{d}/ABC/limit/knapsack_ABC_limit_1_trace.csv", "limit = 10"),
                (f"{d}/ABC/base/knapsack_ABC_base_trace.csv", "limit = 25 (default)"),
                (f"{d}/ABC/limit/knapsack_ABC_limit_2_trace.csv", "limit = 50"),
                (f"{d}/ABC/limit/knapsack_ABC_limit_3_trace.csv", "limit = 100"),
            ],
            "prefix": "knapsack_abc_convergence_limit",
        },
    }


# ── Shared plotting logic ────────────────────────────────────────────────────

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
    "lines.linewidth": 1.8,
})


def load_and_combine(file_paths, labels, add_norm_fn, algo):
    """Load trace CSVs, add norm_evals column, and concatenate."""
    frames = []
    for path, label in zip(file_paths, labels):
        df = pd.read_csv(path)
        df["label"] = label
        df = add_norm_fn(df, algo)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_stats(combined_df: pd.DataFrame, x_col: str, n_grid: int = 500):
    """Compute median + IQR per label for convergence plotting."""
    stats = {}
    for label, grp in combined_df.groupby("label"):
        curves = []
        for _, run_df in grp.groupby("run_id"):
            run_df = run_df.sort_values(x_col)
            mask = np.isfinite(run_df["best_cost"].values)
            xv = run_df[x_col].values[mask]
            yv = run_df["best_cost"].values[mask]
            if len(xv) >= 2:
                curves.append((xv, yv))

        if not curves:
            continue

        run_maxes = [c[0][-1] for c in curves]
        run_mins = [c[0][0] for c in curves]
        grid_min = max(run_mins)
        grid_max = min(run_maxes)
        if grid_max <= grid_min:
            grid_max = max(run_maxes)
            grid_min = min(run_mins)
        grid = np.linspace(grid_min, grid_max, n_grid)

        interp = np.empty((len(curves), n_grid))
        for i, (xv, yv) in enumerate(curves):
            interp[i] = np.interp(grid, xv, yv)

        med = np.median(interp, axis=0)
        q25 = np.quantile(interp, 0.25, axis=0)
        q75 = np.quantile(interp, 0.75, axis=0)
        stats[label] = (grid, med, q25, q75)
    return stats


def plot_convergence(stats, title, xlabel, out_path, labels_order=None):
    fig, ax = plt.subplots()
    if labels_order is None:
        labels_order = list(stats.keys())

    for i, label in enumerate(labels_order):
        if label not in stats:
            continue
        x, med, q25, q75 = stats[label]
        color = COLORS[i % len(COLORS)]
        ax.plot(x, med, color=color, label=label)
        ax.fill_between(x, q25, q75, color=color, alpha=0.15)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Best Cost")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot parameter sensitivity convergence curves"
    )
    p.add_argument(
        "--problem",
        type=str,
        required=True,
        choices=["TSP", "Knapsack", "GraphColoring"],
        help="Problem type to plot: TSP, Knapsack, or GraphColoring",
    )
    p.add_argument(
        "--exp",
        nargs="*",
        default=None,
        help="Optional experiment keys to plot (default: all)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    problem = args.problem
    sens_dir = _sens_dir(problem)
    out_dir = _out_dir(problem)

    if problem == "TSP":
        experiments = _tsp_experiments(sens_dir)
        add_norm_fn = _add_norm_evals_tsp
    elif problem == "Knapsack":
        experiments = _knapsack_experiments(sens_dir)
        add_norm_fn = _add_norm_evals_knapsack
    else:  # GraphColoring
        experiments = _gc_experiments(sens_dir)
        add_norm_fn = _add_norm_evals_gc

    print(f"Generating parameter sensitivity plots for {problem}...\n")

    if args.exp:
        selected = [k for k in args.exp if k in experiments]
        missing = [k for k in args.exp if k not in experiments]
        if missing:
            print(f"Warning: unknown experiment key(s): {missing}")
            print(f"Available keys: {list(experiments.keys())}")
    else:
        selected = list(experiments.keys())

    for exp_name in selected:
        cfg = experiments[exp_name]
        algo = cfg["algo"]
        param = cfg["param"]
        prefix = cfg["prefix"]

        paths = [p for p, _ in cfg["files"]]
        labels = [l for _, l in cfg["files"]]
        missing_paths = [p for p in paths if not os.path.exists(p)]
        if missing_paths:
            print(f"SKIP {exp_name}: missing files")
            for p in missing_paths:
                print(f"  - {p}")
            continue

        print(f"[{exp_name}] {algo} - {param}")
        combined = load_and_combine(paths, labels, add_norm_fn, algo)

        # Raw evals plot
        stats_raw = compute_stats(combined, "evals_cost")
        plot_convergence(
            stats_raw,
            title=f"{algo} convergence - {param} sensitivity ({problem})",
            xlabel="Evaluations (raw evals_cost)",
            out_path=os.path.join(out_dir, f"{prefix}.pdf"),
            labels_order=labels,
        )

        # Normalized evals plot
        stats_norm = compute_stats(combined, "norm_evals")
        plot_convergence(
            stats_norm,
            title=f"{algo} convergence - {param} sensitivity ({problem}, normalized)",
            xlabel="Normalized Evaluations",
            out_path=os.path.join(out_dir, f"{prefix}_normalized.pdf"),
            labels_order=labels,
        )

    print(f"\nDone! Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
