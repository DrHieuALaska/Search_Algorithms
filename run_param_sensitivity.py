"""
run_param_sensitivity.py
========================
Run parameter sensitivity experiments for 3 problems:
1) TSP (manual one-factor-at-a-time sweeps)
2) Knapsack (manual one-factor-at-a-time sweeps)
3) Graph Coloring (config + runner)

Writes config(s), runs experiment runner(s), and moves output CSVs to per-problem
folders inside `src/parameter sensitivity`.

Usage:
    .venv/bin/python3 run_param_sensitivity.py
    .venv/bin/python3 run_param_sensitivity.py --problem tsp
    .venv/bin/python3 run_param_sensitivity.py --problem knapsack graphcolor
    .venv/bin/python3 run_param_sensitivity.py --problem tsp --algo GA TLBO
    .venv/bin/python3 run_param_sensitivity.py --problem graphcolor --algo SA ACO
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import yaml

ROOT      = os.path.dirname(os.path.abspath(__file__))
TSP_CONFIG    = os.path.join(ROOT, "configs", "config_tsp.yaml")
TSP_RUNNER    = os.path.join(ROOT, "src", "experiments", "run_tsp.py")
KP_CONFIG     = os.path.join(ROOT, "configs", "config_knapsack.yaml")
KP_RUNNER     = os.path.join(ROOT, "src", "experiments", "run_knapsack.py")
GC_CONFIG     = os.path.join(ROOT, "configs", "config_graphcolor.yaml")
GC_RUNNER     = os.path.join(ROOT, "src", "experiments", "run_graph_coloring.py")
DEST_BASE = os.path.join(ROOT, "src", "parameter sensitivity")

# Prefer project venv Python, fallback to current interpreter.
PYTHON = os.path.join(ROOT, ".venv", "bin", "python3")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable

# ── Experiment settings (shared across all runs) ────────────────────────────
EXPERIMENT = {
    "exp_id":            "exp_001",
    "n_list":            [30],
    "instances":         5,
    "runs_per_instance": 3,
    "scale":             100.0,
    "code_version":      "",
}

# ── Base (default) configs per algorithm ────────────────────────────────────
SA_BASE = {
    "T0": 100, "Tmin": 0.001, "alpha": 0.99,
    "steps_per_temp": 2000, "max_iter": 50000, "trace_every": 200,
}

GA_BASE = {
    "pop": 100, "crossover": 0.8, "mutation_rate": 0.02,
    "tournament_k": 5, "generation": "(50k - pop)/pop",
}

HC_BASE = {
    "max_iter": 50000, "mode": "first", "trace_every": 200,
}

TLBO_BASE = {
    "pop_size": 50, "move_frac_max": 0.4, "iter": 500,
}

ACO_BASE = {
    "n_ants": 30, "iters": 1667, "alpha": 1.0, "beta": 2.0,
    "rho": 0.2, "Q": 100.0, "trace_every": 10, "deposit_best_only": False,
}

KP_EXPERIMENT = {
    "exp_id": "exp_kp_001",
    "n_items_list": [100],
    "instances": 5,
    "runs_per_instance": 3,
    "weight_low": 1,
    "weight_high": 50,
    "value_low": 1,
    "value_high": 100,
    "capacity_ratio": 0.5,
    "budget_evals": 50000,
    "penalty_lambda": 1000.0,
    "feasible_only": True,
    "code_version": "",
}

KP_SA_BASE = {
    "T0": 100.0,
    "Tmin": 0.001,
    "alpha": 0.99,
    "steps_per_temp": 2000,
    "trace_every": 200,
}

KP_GA_BASE = {
    "population_size": 100,
    "crossover_rate": 0.8,
    "mutation_rate": 0.02,
    "tournament_k": 5,
    "trace_every": 10,
}

KP_HC_BASE = {
    "mode": "first",
    "flip_k": 1,
    "trace_every": 200,
}

KP_TLBO_BASE = {
    "pop_size": 50,
    "move_frac_max": 0.4,
    "iters": 500,
    "trace_every": 10,
}

KP_ABC_BASE = {
    "sn": 50,
    "limit": 25,
    "trace_every": 10,
}

GC_EXPERIMENT = {
    "exp_id": "exp_gc_001",
    "n_nodes_list": [30],
    "n_colors": 4,
    "edge_prob": 0.2,
    "ensure_connected": False,
    "instances": 5,
    "runs_per_instance": 3,
    "budget_evals": 50000,
    "code_version": "",
}

GC_SA_BASE = {
    "T0": 5.0,
    "Tmin": 0.001,
    "alpha": 0.99,
    "trace_every": 200,
}

GC_GA_BASE = {
    "population_size": 100,
    "crossover_rate": 0.9,
    "mutation_rate": 0.02,
    "tournament_k": 10,
    "trace_every": 10,
}

GC_HC_BASE = {
    "mode": "first",
    "trace_every": 200,
}

GC_ACO_BASE = {
    "n_ants": 30,
    "alpha": 1.0,
    "beta": 2.0,
    "rho": 0.1,
    "Q": 1.0,
    "trace_every": 10,
    "deposit_best_only": True,
}

GC_DFS_BASE = {
    "time_limit_sec": 2.0,
    "max_backtracks": 2000000,
    "trace_every": 10000,
}

TSP_ALGO_MAP = {
    "SA": "SA",
    "GA": "GA",
    "HC": "HC",
    "TLBO": "TLBO",
    "ACO": "ACO",
}

KNAPSACK_ALGO_MAP = {
    "SA": "SA_KP",
    "GA": "GA_KP",
    "HC": "HC_KP",
    "TLBO": "TLBO_KP",
    "ABC": "ABC_KP",
}

GC_ALGO_MAP = {
    "SA": "SA_GC",
    "GA": "GA_GC",
    "HC": "HC_GC",
    "ACO": "ACO_GC",
    "DFS": "DFS_GC",
}

# ── All experiments ─────────────────────────────────────────────────────────
# Format: (algo, algo_base_cfg, folder, file_tag, overrides)
EXPERIMENTS = [
    # ── SA ───────────────────────────────────────────────────────────────
    # base
    ("SA", SA_BASE, "TSP/SA/base",           "SA_base",   {}),
    # T0
    ("SA", SA_BASE, "TSP/SA/T0",             "SA_T0_1",   {"T0": 10}),
    ("SA", SA_BASE, "TSP/SA/T0",             "SA_T0_2",   {"T0": 500}),
    ("SA", SA_BASE, "TSP/SA/T0",             "SA_T0_3",   {"T0": 1000}),
    # steps_per_temp
    ("SA", SA_BASE, "TSP/SA/steps_per_temp", "SA_step_1", {"steps_per_temp": 500}),
    ("SA", SA_BASE, "TSP/SA/steps_per_temp", "SA_step_2", {"steps_per_temp": 1000}),
    ("SA", SA_BASE, "TSP/SA/steps_per_temp", "SA_step_3", {"steps_per_temp": 4000}),

    # ── GA ───────────────────────────────────────────────────────────────
    # base
    ("GA", GA_BASE, "TSP/GA/base",           "GA_base",   {}),
    # pop
    ("GA", GA_BASE, "TSP/GA/pop",            "GA_pop_1",  {"pop": 20}),
    ("GA", GA_BASE, "TSP/GA/pop",            "GA_pop_2",  {"pop": 50}),
    ("GA", GA_BASE, "TSP/GA/pop",            "GA_pop_3",  {"pop": 500}),
    # crossover
    ("GA", GA_BASE, "TSP/GA/crossover",      "GA_cross_1",    {"crossover": 0.6}),
    ("GA", GA_BASE, "TSP/GA/crossover",      "GA_cross_2",    {"crossover": 0.7}),
    ("GA", GA_BASE, "TSP/GA/crossover",      "GA_cross_base", {"crossover": 0.8}),
    ("GA", GA_BASE, "TSP/GA/crossover",      "GA_cross_3",    {"crossover": 0.9}),
    # mutation_rate
    ("GA", GA_BASE, "TSP/GA/mutation_rate",  "GA_muta_1", {"mutation_rate": 0.01}),
    ("GA", GA_BASE, "TSP/GA/mutation_rate",  "GA_muta_2", {"mutation_rate": 0.05}),
    ("GA", GA_BASE, "TSP/GA/mutation_rate",  "GA_muta_3", {"mutation_rate": 0.10}),
    # tournament_k
    ("GA", GA_BASE, "TSP/GA/tournament_k",   "GA_tour_1", {"tournament_k": 3}),
    ("GA", GA_BASE, "TSP/GA/tournament_k",   "GA_tour_2", {"tournament_k": 10}),
    ("GA", GA_BASE, "TSP/GA/tournament_k",   "GA_tour_3", {"tournament_k": 20}),

    # ── HC ───────────────────────────────────────────────────────────────
    ("HC", HC_BASE, "TSP/HC/best",           "HC_best",   {"mode": "best"}),
    ("HC", HC_BASE, "TSP/HC/first",          "HC_first",  {"mode": "first"}),

    # ── TLBO ─────────────────────────────────────────────────────────────
    # base
    ("TLBO", TLBO_BASE, "TSP/TLBO/base",           "TLBO_base",   {}),
    # pop_size
    ("TLBO", TLBO_BASE, "TSP/TLBO/pop_size",       "TLBO_pop_1",  {"pop_size": 30}),
    ("TLBO", TLBO_BASE, "TSP/TLBO/pop_size",       "TLBO_pop_2",  {"pop_size": 80}),
    ("TLBO", TLBO_BASE, "TSP/TLBO/pop_size",       "TLBO_pop_3",  {"pop_size": 120}),
    # move_frac_max
    ("TLBO", TLBO_BASE, "TSP/TLBO/move_frac_max",  "TLBO_move_1", {"move_frac_max": 0.2}),
    ("TLBO", TLBO_BASE, "TSP/TLBO/move_frac_max",  "TLBO_move_2", {"move_frac_max": 0.6}),
    ("TLBO", TLBO_BASE, "TSP/TLBO/move_frac_max",  "TLBO_move_3", {"move_frac_max": 0.8}),

    # ── ACO ──────────────────────────────────────────────────────────────
    # base (used for n_ants default)
    ("ACO", ACO_BASE, "TSP/ACO/base",            "ACO_base",       {}),
    # n_ants
    ("ACO", ACO_BASE, "TSP/ACO/n_ants",            "ACO_nants_1",    {"n_ants": 10, "iters": 5000}),
    ("ACO", ACO_BASE, "TSP/ACO/n_ants",            "ACO_nants_2",    {"n_ants": 20, "iters": 2500}),
    ("ACO", ACO_BASE, "TSP/ACO/n_ants",            "ACO_nants_3",    {"n_ants": 50, "iters": 1000}),
    # alpha
    ("ACO", ACO_BASE, "TSP/ACO/alpha",             "ACO_alpha_1",    {"alpha": 0.5}),
    ("ACO", ACO_BASE, "TSP/ACO/alpha",             "ACO_alpha_2",    {"alpha": 2.0}),
    ("ACO", ACO_BASE, "TSP/ACO/alpha",             "ACO_alpha_3",    {"alpha": 3.0}),
    # beta
    ("ACO", ACO_BASE, "TSP/ACO/beta",              "ACO_beta_1",     {"beta": 1.0}),
    ("ACO", ACO_BASE, "TSP/ACO/beta",              "ACO_beta_2",     {"beta": 3.0}),
    ("ACO", ACO_BASE, "TSP/ACO/beta",              "ACO_beta_3",     {"beta": 5.0}),
    # rho
    ("ACO", ACO_BASE, "TSP/ACO/rho",               "ACO_rho_1",      {"rho": 0.05}),
    ("ACO", ACO_BASE, "TSP/ACO/rho",               "ACO_rho_2",      {"rho": 0.1}),
    ("ACO", ACO_BASE, "TSP/ACO/rho",               "ACO_rho_3",      {"rho": 0.5}),
]

GC_EXPERIMENTS = [
    # ── SA_GC ─────────────────────────────────────────────────────────────
    # base
    ("SA_GC", GC_SA_BASE, "GraphColoring/SA/base",         "GC_SA_base",    {}),
    # T0
    ("SA_GC", GC_SA_BASE, "GraphColoring/SA/T0",           "GC_SA_T0_1",    {"T0": 1.0}),
    ("SA_GC", GC_SA_BASE, "GraphColoring/SA/T0",           "GC_SA_T0_2",    {"T0": 10.0}),
    ("SA_GC", GC_SA_BASE, "GraphColoring/SA/T0",           "GC_SA_T0_3",    {"T0": 50.0}),
    # alpha
    ("SA_GC", GC_SA_BASE, "GraphColoring/SA/alpha",        "GC_SA_alpha_1", {"alpha": 0.95}),
    ("SA_GC", GC_SA_BASE, "GraphColoring/SA/alpha",        "GC_SA_alpha_2", {"alpha": 0.97}),
    ("SA_GC", GC_SA_BASE, "GraphColoring/SA/alpha",        "GC_SA_alpha_3", {"alpha": 0.999}),

    # ── GA_GC ─────────────────────────────────────────────────────────────
    # base
    ("GA_GC", GC_GA_BASE, "GraphColoring/GA/base",         "GC_GA_base",    {}),
    # population_size
    ("GA_GC", GC_GA_BASE, "GraphColoring/GA/pop",          "GC_GA_pop_1",   {"population_size": 20}),
    ("GA_GC", GC_GA_BASE, "GraphColoring/GA/pop",          "GC_GA_pop_2",   {"population_size": 50}),
    ("GA_GC", GC_GA_BASE, "GraphColoring/GA/pop",          "GC_GA_pop_3",   {"population_size": 200}),
    # crossover_rate
    ("GA_GC", GC_GA_BASE, "GraphColoring/GA/crossover",    "GC_GA_cross_1", {"crossover_rate": 0.6}),
    ("GA_GC", GC_GA_BASE, "GraphColoring/GA/crossover",    "GC_GA_cross_2", {"crossover_rate": 0.7}),
    ("GA_GC", GC_GA_BASE, "GraphColoring/GA/crossover",    "GC_GA_cross_3", {"crossover_rate": 0.95}),
    # mutation_rate
    ("GA_GC", GC_GA_BASE, "GraphColoring/GA/mutation_rate","GC_GA_muta_1",  {"mutation_rate": 0.01}),
    ("GA_GC", GC_GA_BASE, "GraphColoring/GA/mutation_rate","GC_GA_muta_2",  {"mutation_rate": 0.05}),
    ("GA_GC", GC_GA_BASE, "GraphColoring/GA/mutation_rate","GC_GA_muta_3",  {"mutation_rate": 0.10}),
    # tournament_k
    ("GA_GC", GC_GA_BASE, "GraphColoring/GA/tournament_k", "GC_GA_tour_1",  {"tournament_k": 3}),
    ("GA_GC", GC_GA_BASE, "GraphColoring/GA/tournament_k", "GC_GA_tour_2",  {"tournament_k": 5}),
    ("GA_GC", GC_GA_BASE, "GraphColoring/GA/tournament_k", "GC_GA_tour_3",  {"tournament_k": 20}),

    # ── HC_GC ─────────────────────────────────────────────────────────────
    ("HC_GC", GC_HC_BASE, "GraphColoring/HC/first",        "GC_HC_first",   {"mode": "first"}),
    ("HC_GC", GC_HC_BASE, "GraphColoring/HC/best",         "GC_HC_best",    {"mode": "best"}),

    # ── ACO_GC ────────────────────────────────────────────────────────────
    # base
    ("ACO_GC", GC_ACO_BASE, "GraphColoring/ACO/base",      "GC_ACO_base",    {}),
    # n_ants
    ("ACO_GC", GC_ACO_BASE, "GraphColoring/ACO/n_ants",    "GC_ACO_nants_1", {"n_ants": 10}),
    ("ACO_GC", GC_ACO_BASE, "GraphColoring/ACO/n_ants",    "GC_ACO_nants_2", {"n_ants": 20}),
    ("ACO_GC", GC_ACO_BASE, "GraphColoring/ACO/n_ants",    "GC_ACO_nants_3", {"n_ants": 50}),
    # alpha
    ("ACO_GC", GC_ACO_BASE, "GraphColoring/ACO/alpha",     "GC_ACO_alpha_1", {"alpha": 0.5}),
    ("ACO_GC", GC_ACO_BASE, "GraphColoring/ACO/alpha",     "GC_ACO_alpha_2", {"alpha": 2.0}),
    ("ACO_GC", GC_ACO_BASE, "GraphColoring/ACO/alpha",     "GC_ACO_alpha_3", {"alpha": 3.0}),
    # beta
    ("ACO_GC", GC_ACO_BASE, "GraphColoring/ACO/beta",      "GC_ACO_beta_1",  {"beta": 1.0}),
    ("ACO_GC", GC_ACO_BASE, "GraphColoring/ACO/beta",      "GC_ACO_beta_2",  {"beta": 3.0}),
    ("ACO_GC", GC_ACO_BASE, "GraphColoring/ACO/beta",      "GC_ACO_beta_3",  {"beta": 5.0}),
    # rho
    ("ACO_GC", GC_ACO_BASE, "GraphColoring/ACO/rho",       "GC_ACO_rho_1",   {"rho": 0.05}),
    ("ACO_GC", GC_ACO_BASE, "GraphColoring/ACO/rho",       "GC_ACO_rho_2",   {"rho": 0.2}),
    ("ACO_GC", GC_ACO_BASE, "GraphColoring/ACO/rho",       "GC_ACO_rho_3",   {"rho": 0.5}),

    # ── DFS_GC ─────────────────────────────────────────────────────────
    # base
    ("DFS_GC", GC_DFS_BASE, "GraphColoring/DFS/base",       "GC_DFS_base",         {}),
    # time_limit_sec
    ("DFS_GC", GC_DFS_BASE, "GraphColoring/DFS/time_limit", "GC_DFS_time_1",       {"time_limit_sec": 0.5}),
    ("DFS_GC", GC_DFS_BASE, "GraphColoring/DFS/time_limit", "GC_DFS_time_2",       {"time_limit_sec": 1.0}),
    ("DFS_GC", GC_DFS_BASE, "GraphColoring/DFS/time_limit", "GC_DFS_time_3",       {"time_limit_sec": 5.0}),
    # max_backtracks
    ("DFS_GC", GC_DFS_BASE, "GraphColoring/DFS/max_backtracks", "GC_DFS_bt_1",     {"max_backtracks": 100000}),
    ("DFS_GC", GC_DFS_BASE, "GraphColoring/DFS/max_backtracks", "GC_DFS_bt_2",     {"max_backtracks": 500000}),
    ("DFS_GC", GC_DFS_BASE, "GraphColoring/DFS/max_backtracks", "GC_DFS_bt_3",     {"max_backtracks": 5000000}),
]

KP_EXPERIMENTS = [
    # SA_KP
    ("SA_KP", KP_SA_BASE, "Knapsack/SA/base", "knapsack_SA_base", {}),
    ("SA_KP", KP_SA_BASE, "Knapsack/SA/T0", "knapsack_SA_T0_1", {"T0": 10.0}),
    ("SA_KP", KP_SA_BASE, "Knapsack/SA/T0", "knapsack_SA_T0_2", {"T0": 500.0}),
    ("SA_KP", KP_SA_BASE, "Knapsack/SA/T0", "knapsack_SA_T0_3", {"T0": 1000.0}),
    ("SA_KP", KP_SA_BASE, "Knapsack/SA/steps_per_temp", "knapsack_SA_step_1", {"steps_per_temp": 500}),
    ("SA_KP", KP_SA_BASE, "Knapsack/SA/steps_per_temp", "knapsack_SA_step_2", {"steps_per_temp": 1000}),
    ("SA_KP", KP_SA_BASE, "Knapsack/SA/steps_per_temp", "knapsack_SA_step_3", {"steps_per_temp": 4000}),

    # GA_KP
    ("GA_KP", KP_GA_BASE, "Knapsack/GA/base", "knapsack_GA_base", {}),
    ("GA_KP", KP_GA_BASE, "Knapsack/GA/pop", "knapsack_GA_pop_1", {"population_size": 20}),
    ("GA_KP", KP_GA_BASE, "Knapsack/GA/pop", "knapsack_GA_pop_2", {"population_size": 50}),
    ("GA_KP", KP_GA_BASE, "Knapsack/GA/pop", "knapsack_GA_pop_3", {"population_size": 500}),
    ("GA_KP", KP_GA_BASE, "Knapsack/GA/crossover", "knapsack_GA_cross_1", {"crossover_rate": 0.6}),
    ("GA_KP", KP_GA_BASE, "Knapsack/GA/crossover", "knapsack_GA_cross_2", {"crossover_rate": 0.7}),
    ("GA_KP", KP_GA_BASE, "Knapsack/GA/crossover", "knapsack_GA_cross_3", {"crossover_rate": 0.9}),
    ("GA_KP", KP_GA_BASE, "Knapsack/GA/mutation_rate", "knapsack_GA_mut_1", {"mutation_rate": 0.01}),
    ("GA_KP", KP_GA_BASE, "Knapsack/GA/mutation_rate", "knapsack_GA_mut_2", {"mutation_rate": 0.05}),
    ("GA_KP", KP_GA_BASE, "Knapsack/GA/mutation_rate", "knapsack_GA_mut_3", {"mutation_rate": 0.10}),
    ("GA_KP", KP_GA_BASE, "Knapsack/GA/tournament_k", "knapsack_GA_tour_1", {"tournament_k": 3}),
    ("GA_KP", KP_GA_BASE, "Knapsack/GA/tournament_k", "knapsack_GA_tour_2", {"tournament_k": 10}),
    ("GA_KP", KP_GA_BASE, "Knapsack/GA/tournament_k", "knapsack_GA_tour_3", {"tournament_k": 20}),

    # HC_KP
    ("HC_KP", KP_HC_BASE, "Knapsack/HC/first", "knapsack_HC_first", {"mode": "first"}),
    ("HC_KP", KP_HC_BASE, "Knapsack/HC/best", "knapsack_HC_best", {"mode": "best"}),

    # TLBO_KP
    ("TLBO_KP", KP_TLBO_BASE, "Knapsack/TLBO/base", "knapsack_TLBO_base", {}),
    ("TLBO_KP", KP_TLBO_BASE, "Knapsack/TLBO/pop_size", "knapsack_TLBO_pop_1", {"pop_size": 30}),
    ("TLBO_KP", KP_TLBO_BASE, "Knapsack/TLBO/pop_size", "knapsack_TLBO_pop_2", {"pop_size": 80}),
    ("TLBO_KP", KP_TLBO_BASE, "Knapsack/TLBO/pop_size", "knapsack_TLBO_pop_3", {"pop_size": 120}),
    ("TLBO_KP", KP_TLBO_BASE, "Knapsack/TLBO/move_frac_max", "knapsack_TLBO_move_1", {"move_frac_max": 0.2}),
    ("TLBO_KP", KP_TLBO_BASE, "Knapsack/TLBO/move_frac_max", "knapsack_TLBO_move_2", {"move_frac_max": 0.6}),
    ("TLBO_KP", KP_TLBO_BASE, "Knapsack/TLBO/move_frac_max", "knapsack_TLBO_move_3", {"move_frac_max": 0.8}),

    # ABC_KP
    ("ABC_KP", KP_ABC_BASE, "Knapsack/ABC/base", "knapsack_ABC_base", {}),
    ("ABC_KP", KP_ABC_BASE, "Knapsack/ABC/sn", "knapsack_ABC_sn_1", {"sn": 10}),
    ("ABC_KP", KP_ABC_BASE, "Knapsack/ABC/sn", "knapsack_ABC_sn_2", {"sn": 100}),
    ("ABC_KP", KP_ABC_BASE, "Knapsack/ABC/sn", "knapsack_ABC_sn_3", {"sn": 500}),
    ("ABC_KP", KP_ABC_BASE, "Knapsack/ABC/limit", "knapsack_ABC_limit_1", {"limit": 10}),
    ("ABC_KP", KP_ABC_BASE, "Knapsack/ABC/limit", "knapsack_ABC_limit_2", {"limit": 50}),
    ("ABC_KP", KP_ABC_BASE, "Knapsack/ABC/limit", "knapsack_ABC_limit_3", {"limit": 100}),
]


def build_config(algo, algo_cfg, file_tag):
    """Build the full YAML config dict for one experiment."""
    cfg = {"experiment": dict(EXPERIMENT)}
    cfg["experiment"]["run_csv"]   = f"{file_tag}_run.csv"
    cfg["experiment"]["trace_csv"] = f"{file_tag}_trace.csv"
    cfg["algos"] = [algo]
    cfg[algo] = dict(algo_cfg)
    return cfg


def build_config_knapsack(algo, algo_cfg, file_tag):
    cfg = {"experiment": dict(KP_EXPERIMENT)}
    cfg["experiment"]["run_csv"] = f"{file_tag}_run.csv"
    cfg["experiment"]["trace_csv"] = f"{file_tag}_trace.csv"
    cfg["algos"] = [algo]
    cfg[algo] = dict(algo_cfg)
    return cfg


def build_config_gc(algo, algo_cfg, file_tag):
    cfg = {"experiment": dict(GC_EXPERIMENT)}
    cfg["experiment"]["run_csv"] = f"{file_tag}_run.csv"
    cfg["experiment"]["trace_csv"] = f"{file_tag}_trace.csv"
    cfg["algos"] = [algo]
    cfg[algo] = dict(algo_cfg)
    return cfg


def run_one(algo, algo_base, folder, file_tag, overrides, idx, total):
    """Run a single experiment: write config, execute, move outputs."""
    dest_dir = os.path.join(DEST_BASE, folder)
    os.makedirs(dest_dir, exist_ok=True)

    # Check if already done
    run_csv   = f"{file_tag}_run.csv"
    trace_csv = f"{file_tag}_trace.csv"
    if (os.path.exists(os.path.join(dest_dir, run_csv))
            and os.path.exists(os.path.join(dest_dir, trace_csv))):
        param_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "default"
        print(f"[{idx}/{total}] SKIP {file_tag} ({param_str}) — already exists")
        return True

    # Build config
    algo_cfg = dict(algo_base)
    algo_cfg.update(overrides)
    cfg = build_config(algo, algo_cfg, file_tag)

    with open(TSP_CONFIG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    param_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "default"
    print(f"[{idx}/{total}] {file_tag}: {param_str} ...", end=" ", flush=True)

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, TSP_RUNNER, "--config", TSP_CONFIG],
        capture_output=True, text=True,
    )
    dt = time.time() - t0

    if result.returncode != 0:
        print(f"FAILED ({dt:.1f}s)")
        print(f"  stderr: {result.stderr[:300]}")
        return False

    # Move outputs
    for csv in [run_csv, trace_csv]:
        src = os.path.join(ROOT, csv)
        if os.path.exists(src):
            shutil.move(src, os.path.join(dest_dir, csv))

    print(f"OK ({dt:.1f}s)")
    return True


def run_one_knapsack(algo, algo_base, folder, file_tag, overrides, idx, total):
    dest_dir = os.path.join(DEST_BASE, folder)
    os.makedirs(dest_dir, exist_ok=True)

    run_csv = f"{file_tag}_run.csv"
    trace_csv = f"{file_tag}_trace.csv"
    if (os.path.exists(os.path.join(dest_dir, run_csv))
            and os.path.exists(os.path.join(dest_dir, trace_csv))):
        param_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "default"
        print(f"[{idx}/{total}] SKIP {file_tag} ({param_str}) — already exists")
        return True

    algo_cfg = dict(algo_base)
    algo_cfg.update(overrides)
    cfg = build_config_knapsack(algo, algo_cfg, file_tag)

    with open(KP_CONFIG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    param_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "default"
    print(f"[{idx}/{total}] {file_tag}: {param_str} ...", end=" ", flush=True)

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, KP_RUNNER, "--config", KP_CONFIG],
        capture_output=True,
        text=True,
    )
    dt = time.time() - t0

    if result.returncode != 0:
        print(f"FAILED ({dt:.1f}s)")
        print(f"  stderr: {result.stderr[:500]}")
        return False

    for csv in [run_csv, trace_csv]:
        src = os.path.join(ROOT, csv)
        if os.path.exists(src):
            shutil.move(src, os.path.join(dest_dir, csv))

    print(f"OK ({dt:.1f}s)")
    return True


def run_one_gc(algo, algo_base, folder, file_tag, overrides, idx, total):
    dest_dir = os.path.join(DEST_BASE, folder)
    os.makedirs(dest_dir, exist_ok=True)

    run_csv   = f"{file_tag}_run.csv"
    trace_csv = f"{file_tag}_trace.csv"
    if (os.path.exists(os.path.join(dest_dir, run_csv))
            and os.path.exists(os.path.join(dest_dir, trace_csv))):
        param_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "default"
        print(f"[{idx}/{total}] SKIP {file_tag} ({param_str}) — already exists")
        return True

    algo_cfg = dict(algo_base)
    algo_cfg.update(overrides)
    cfg = build_config_gc(algo, algo_cfg, file_tag)

    with open(GC_CONFIG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    param_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "default"
    print(f"[{idx}/{total}] {file_tag}: {param_str} ...", end=" ", flush=True)

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, GC_RUNNER, "--config", GC_CONFIG],
        capture_output=True,
        text=True,
    )
    dt = time.time() - t0

    if result.returncode != 0:
        print(f"FAILED ({dt:.1f}s)")
        print(f"  stderr: {result.stderr[:500]}")
        return False

    for csv in [run_csv, trace_csv]:
        src = os.path.join(ROOT, csv)
        if os.path.exists(src):
            shutil.move(src, os.path.join(dest_dir, csv))

    print(f"OK ({dt:.1f}s)")
    return True


def run_problem_with_config(
    label,
    config_path,
    runner_path,
    problem_subdir,
    override_algos=None,
    force=False,
):
    """Run one config-driven problem and move run/trace CSV outputs."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    effective_config = config_path
    temp_config = None
    if override_algos:
        cfg["algos"] = list(override_algos)
        fd, temp_config = tempfile.mkstemp(prefix="param_sensitivity_", suffix=".yaml", dir=ROOT)
        os.close(fd)
        with open(temp_config, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        effective_config = temp_config

    exp = cfg.get("experiment", {})
    run_csv = exp.get("run_csv", "runs.csv")
    trace_csv = exp.get("trace_csv", "trace.csv")

    dest_dir = os.path.join(DEST_BASE, problem_subdir)
    os.makedirs(dest_dir, exist_ok=True)

    run_dest = os.path.join(dest_dir, run_csv)
    trace_dest = os.path.join(dest_dir, trace_csv)

    if force:
        for p in [run_dest, trace_dest]:
            if os.path.exists(p):
                os.remove(p)

    if (not force) and os.path.exists(run_dest) and os.path.exists(trace_dest):
        print(f"[SKIP] {label} — output already exists")
        return True, True

    print(f"[RUN ] {label} ...", end=" ", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, runner_path, "--config", effective_config],
        capture_output=True,
        text=True,
    )
    dt = time.time() - t0

    if result.returncode != 0:
        if temp_config and os.path.exists(temp_config):
            os.remove(temp_config)
        print(f"FAILED ({dt:.1f}s)")
        print(f"  stderr: {result.stderr[:500]}")
        return False, False

    for csv_name in [run_csv, trace_csv]:
        src = os.path.join(ROOT, csv_name)
        dest = os.path.join(dest_dir, csv_name)
        if os.path.exists(src):
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(src, dest)

    if temp_config and os.path.exists(temp_config):
        os.remove(temp_config)

    print(f"OK ({dt:.1f}s)")
    return True, False


def main():
    parser = argparse.ArgumentParser(
        description="Run parameter sensitivity experiments for TSP, Knapsack, and Graph Coloring"
    )
    parser.add_argument(
        "--problem",
        nargs="*",
        default=None,
        help="Filter by problem(s): tsp knapsack graphcolor (default: all)",
    )
    parser.add_argument(
        "--algo", nargs="*", default=None,
        help="Filter algorithm(s) — TSP: SA GA HC TLBO ACO | Knapsack: SA GA HC TLBO ABC | GraphColoring: SA GA HC ACO DFS",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if output CSVs already exist",
    )
    args = parser.parse_args()

    allowed_problems = {"tsp", "knapsack", "graphcolor"}
    if args.problem:
        selected_problems = {p.lower() for p in args.problem}
        invalid = selected_problems - allowed_problems
        if invalid:
            print(f"Invalid --problem value(s): {', '.join(sorted(invalid))}")
            print("Allowed: tsp knapsack graphcolor")
            sys.exit(2)
    else:
        selected_problems = set(allowed_problems)

    total_ok = 0
    total_skip = 0
    total_fail = 0
    overall_start = time.time()

    # ── TSP (manual sweep list in this file) ───────────────────────────────
    if "tsp" in selected_problems:
        print("\n[TSP] Manual parameter sweep")

        # Filter experiments
        if args.algo:
            allowed = {a.upper() for a in args.algo}
            invalid = sorted(a for a in allowed if a not in set(TSP_ALGO_MAP.keys()))
            if invalid:
                print(f"Invalid TSP --algo value(s): {', '.join(invalid)}")
                print("Allowed for TSP: SA GA HC TLBO ACO")
                sys.exit(2)
            allowed_tsp = {TSP_ALGO_MAP[a] for a in allowed}
            exps = [(a, b, f, t, o) for a, b, f, t, o in EXPERIMENTS if a in allowed_tsp]
        else:
            exps = list(EXPERIMENTS)

        total = len(exps)
        print(f"TSP experiments: {total}\n")

        # If --force, remove existing outputs first
        if args.force:
            for algo, base_cfg, folder, tag, overrides in exps:
                dest_dir = os.path.join(DEST_BASE, folder)
                for suffix in ["_run.csv", "_trace.csv"]:
                    p = os.path.join(dest_dir, f"{tag}{suffix}")
                    if os.path.exists(p):
                        os.remove(p)

        ok = 0
        skip = 0
        fail = 0

        for i, (algo, base_cfg, folder, tag, overrides) in enumerate(exps, 1):
            dest_dir = os.path.join(DEST_BASE, folder)
            run_csv = f"{tag}_run.csv"
            if (not args.force
                    and os.path.exists(os.path.join(dest_dir, run_csv))):
                param_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "default"
                print(f"[{i}/{total}] SKIP {tag} ({param_str})")
                skip += 1
                continue

            success = run_one(algo, base_cfg, folder, tag, overrides, i, total)
            if success:
                ok += 1
            else:
                fail += 1

        total_ok += ok
        total_skip += skip
        total_fail += fail
        print(f"[TSP] done — {ok} ran, {skip} skipped, {fail} failed")

    # ── Knapsack (config runner) ───────────────────────────────────────────
    if "knapsack" in selected_problems:
        print("\n[Knapsack] Manual parameter sweep")

        if args.algo:
            requested = {a.upper() for a in args.algo}
            invalid = sorted(a for a in requested if a not in set(KNAPSACK_ALGO_MAP.keys()))
            if invalid:
                print(f"Invalid Knapsack --algo value(s): {', '.join(invalid)}")
                print("Allowed for Knapsack: SA GA HC TLBO ABC")
                sys.exit(2)
            allowed_kp = {KNAPSACK_ALGO_MAP[a] for a in requested}
            kp_exps = [(a, b, f, t, o) for a, b, f, t, o in KP_EXPERIMENTS if a in allowed_kp]
        else:
            kp_exps = list(KP_EXPERIMENTS)

        total = len(kp_exps)
        print(f"Knapsack experiments: {total}\n")

        if args.force:
            for algo, base_cfg, folder, tag, overrides in kp_exps:
                dest_dir = os.path.join(DEST_BASE, folder)
                for suffix in ["_run.csv", "_trace.csv"]:
                    p = os.path.join(dest_dir, f"{tag}{suffix}")
                    if os.path.exists(p):
                        os.remove(p)

        ok = 0
        skip = 0
        fail = 0
        for i, (algo, base_cfg, folder, tag, overrides) in enumerate(kp_exps, 1):
            dest_dir = os.path.join(DEST_BASE, folder)
            run_csv = f"{tag}_run.csv"
            if (not args.force
                    and os.path.exists(os.path.join(dest_dir, run_csv))):
                param_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "default"
                print(f"[{i}/{total}] SKIP {tag} ({param_str})")
                skip += 1
                continue

            success = run_one_knapsack(algo, base_cfg, folder, tag, overrides, i, total)
            if success:
                ok += 1
            else:
                fail += 1

        total_ok += ok
        total_skip += skip
        total_fail += fail
        print(f"[Knapsack] done — {ok} ran, {skip} skipped, {fail} failed")

    # ── Graph Coloring (manual sweep) ─────────────────────────────────────
    if "graphcolor" in selected_problems:
        print("\n[GraphColoring] Manual parameter sweep")

        if args.algo:
            requested = {a.upper() for a in args.algo}
            invalid = sorted(a for a in requested if a not in set(GC_ALGO_MAP.keys()))
            if invalid:
                print(f"Invalid GraphColoring --algo value(s): {', '.join(invalid)}")
                print("Allowed for GraphColoring: SA GA HC ACO DFS")
                sys.exit(2)
            allowed_gc = {GC_ALGO_MAP[a] for a in requested}
            gc_exps = [(a, b, f, t, o) for a, b, f, t, o in GC_EXPERIMENTS if a in allowed_gc]
        else:
            gc_exps = list(GC_EXPERIMENTS)

        total = len(gc_exps)
        print(f"GraphColoring experiments: {total}\n")

        if args.force:
            for algo, base_cfg, folder, tag, overrides in gc_exps:
                dest_dir = os.path.join(DEST_BASE, folder)
                for suffix in ["_run.csv", "_trace.csv"]:
                    p = os.path.join(dest_dir, f"{tag}{suffix}")
                    if os.path.exists(p):
                        os.remove(p)

        ok = 0
        skip = 0
        fail = 0
        for i, (algo, base_cfg, folder, tag, overrides) in enumerate(gc_exps, 1):
            dest_dir = os.path.join(DEST_BASE, folder)
            run_csv = f"{tag}_run.csv"
            if (not args.force
                    and os.path.exists(os.path.join(dest_dir, run_csv))):
                param_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "default"
                print(f"[{i}/{total}] SKIP {tag} ({param_str})")
                skip += 1
                continue

            success = run_one_gc(algo, base_cfg, folder, tag, overrides, i, total)
            if success:
                ok += 1
            else:
                fail += 1

        total_ok += ok
        total_skip += skip
        total_fail += fail
        print(f"[GraphColoring] done — {ok} ran, {skip} skipped, {fail} failed")
    elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"Finished in {elapsed:.0f}s — {total_ok} ran, {total_skip} skipped, {total_fail} failed")
    print(f"Output: {DEST_BASE}")


if __name__ == "__main__":
    main()
