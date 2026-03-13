import numpy as np
from Parameter_analysis_code.ABC_parameter_analysis import ABC_run_param_analysis
from Parameter_analysis_code.CuckooSearch_parameter_analysis import CuckooSearch_run_param_analysis
from Parameter_analysis_code.FireFly_parameter_analysis import Fireflies_run_param_analysis
from Parameter_analysis_code.PSO_parameter_analysis import PSO_run_param_analysis
from Parameter_analysis_code.DE_parameter_analysis import DE_run_param_analysis 
from Parameter_analysis_code.GA_parameter_analysis import GA_run_param_analysis
from Parameter_analysis_code.TLBO_parameter_analysis import TLBO_run_param_analysis
import os

BUDGET    = 100_000
N_TRIALS  = 10
DIMENSION = 10

RASTRIGIN_BOUNDS  = [[-5.12, 5.12]] * DIMENSION
ROSENBROCK_BOUNDS = [[-5, 10]]      * DIMENSION
SPHERE_BOUNDS     = [[-5.12, 5.12]] * DIMENSION

def sphere(X):      return np.sum(X**2)
def rosenbrock(X):  return np.sum(100*(X[1:] - X[:-1]**2)**2 + (X[:-1] - 1)**2)
def rastrigin(X):
    return 10 * len(X) + np.sum(X**2 - 10 * np.cos(2 * np.pi * X))

OBJECTIVE_FUNCTIONS = [
    {"name": "sphere",     "func": sphere,     "bounds": SPHERE_BOUNDS,     "f_target": 1e-6},
    {"name": "rastrigin",  "func": rastrigin,  "bounds": RASTRIGIN_BOUNDS,  "f_target": 1.0},
    {"name": "rosenbrock", "func": rosenbrock, "bounds": ROSENBROCK_BOUNDS, "f_target": 1.0},
]

# -----------------------------------------------
# Parameter grids — one dict per parameter to analyze
# -----------------------------------------------
PARAM_GRIDS_FOR_ABC = {
    "COLONY_SIZE": [20, 40, 60, 80, 100],
    "LIMIT":       [20, 50, 100, 150, 200],
    "PHI_RANGE":   [(-0.01, 0.01), (-0.1, 0.1), (-0.5, 0.5), (-1, 1), (-2, 2)],
}

PARAM_GRIDS_FOR_CUCKOOSEARCH = {
    "N_NESTS": [20, 25, 30, 40, 60],
    "pa":     [0.1, 0.25, 0.5, 0.75],
    "alpha":  [0.01, 0.1, 0.5, 1],
}

PARAM_GRIDS_FOR_FIREFLIES = {
    "NUM_FIREFLIES": [20, 25, 30, 40, 60],
    "alpha":        [0.01, 0.1, 0.2, 0.5],
    "beta0":        [0.5, 1.0, 1.5, 2.0],
    "fraction_for_gamma": [0.1, 0.25, 0.5, 0.75, 1.0],
}

PARAM_GRIDS_FOR_PSO = {
    "NUM_PARTICLES": [20, 25, 30, 40, 60],
    "w":             [0.1, 0.3, 0.5, 0.7, 0.9],
    "c1":            [0.5, 1.0, 1.5, 2.0],
    "c2":            [0.5, 1.0, 1.5, 2.0],
}

PARAM_GRIDS_FOR_DE = {
    "POP_SIZE": [20, 40, 80, 100, 200],
    "F":        [0.3, 0.5, 0.7, 0.9, 1.2],
    "CR":       [0.1, 0.3, 0.5, 0.7, 0.9],
}

PARAM_GRIDS_FOR_GA = {
    "pop_size": [20, 40, 80, 100, 200],
    "crossover_rate": [0.5, 0.7, 0.9],
    "mutation_rate": [0.01, 0.05, 0.1],
    "mutation_scale_with_range": [0.05, 0.1, 0.2],
    "elite_ratio": [0.0, 0.1, 0.2],
}

PARAM_GRIDS_FOR_TLBO = {
    "pop_size": [20, 40, 80, 100, 200],
}
# for param_name, param_values in PARAM_GRIDS_FOR_ABC.items():
#     ABC_run_param_analysis(param_name, param_values, OBJECTIVE_FUNCTIONS, n_trials=N_TRIALS, budget=BUDGET)

# for param_name, param_values in PARAM_GRIDS_FOR_CUCKOOSEARCH.items():
#     CuckooSearch_run_param_analysis(param_name, param_values, OBJECTIVE_FUNCTIONS, n_trials=N_TRIALS, budget=BUDGET)

# for param_name, param_values in PARAM_GRIDS_FOR_FIREFLIES.items():
#     Fireflies_run_param_analysis(param_name, param_values, OBJECTIVE_FUNCTIONS, n_trials=N_TRIALS, budget=BUDGET)

# for param_name, param_values in PARAM_GRIDS_FOR_PSO.items():
#     PSO_run_param_analysis(param_name, param_values, OBJECTIVE_FUNCTIONS, n_trials=N_TRIALS, budget=BUDGET)

# for param_name, param_values in PARAM_GRIDS_FOR_DE.items():
#     DE_run_param_analysis(param_name, param_values, OBJECTIVE_FUNCTIONS, n_trials=N_TRIALS, budget=BUDGET)

for param_name, param_values in PARAM_GRIDS_FOR_GA.items():
    GA_run_param_analysis(param_name, param_values, OBJECTIVE_FUNCTIONS, n_trials=N_TRIALS, budget=BUDGET)

# for param_name, param_values in PARAM_GRIDS_FOR_TLBO.items():
#     TLBO_run_param_analysis(param_name, param_values, OBJECTIVE_FUNCTIONS, n_trials=N_TRIALS, budget=BUDGET)