import numpy as np
from Algorithms_with_func_evals.ABC_func_evals import ABC_run_trials_multi_func_to_csv
from Algorithms_with_func_evals.CuckooSearch_func_evals import CuckooSearch_run_trials_multi_func_to_csv
from Algorithms_with_func_evals.FireFly_func_evals import FireFly_run_trials_multi_func_to_csv
from Algorithms_with_func_evals.PSO_func_evals import PSO_run_trials_multi_func_to_csv
from Algorithms_with_func_evals.Differential_evolution_func_evals import DE_run_trials_multi_func_to_csv
from Algorithms_with_func_evals.Hill_climbing_multi_starts_func_evals import Hill_climbing_multi_starts_run_trials_multi_func_to_csv
from Algorithms_with_func_evals.Simulated_annealing_multi_starts_func_evals import Simulated_annealing_multi_starts_run_trials_multi_func_to_csv 
from Algorithms_with_func_evals.TLBO_func_evals import TLBO_run_trials_multi_func_to_csv
from Algorithms_with_func_evals.GA_func_evals import GA_run_trials_multi_func_to_csv
import os


FOLDER_PATH = 'trials_data'
# Create the directory if it doesn't exist
if not os.path.exists(FOLDER_PATH):
    os.makedirs(FOLDER_PATH)

F_TARGET_SPHERE = 1e-6
F_TARGET_RASTRIGIN = 1
F_TARGET_ROSENBROCK = 1

DIMENSION = 10

SPHERE_BOUNDS = [[-5.12, 5.12]] * DIMENSION
ROSENBROCK_BOUNDS = [[-5, 10]] * DIMENSION
RASTRIGIN_BOUNDS = [[-5.12, 5.12]] * DIMENSION

INTERVAL_EVALS = 200

def sphere(X):
    return np.sum(X**2)

def rosenbrock_func(X):
    return np.sum(100*(X[1:] - X[:-1]**2)**2 + (X[:-1] - 1)**2)

def rastrigin(X):
    A = 10
    return A * len(X) + np.sum(X**2 - A * np.cos(2 * np.pi * X))

FUNCTIONS_FOR_ABC = [
    {
        "name":     "sphere",
        "func":     sphere,
        "dimension": DIMENSION,
        "bounds":   SPHERE_BOUNDS,
        "colony_size": 20,
        "max_iter": 5000,
        "limit": 200,
        "phi_range": (-1, 1),
        "f_target": F_TARGET_SPHERE,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rastrigin",
        "func":     rastrigin,
        "dimension": DIMENSION,
        "bounds":   RASTRIGIN_BOUNDS,
        "colony_size": 20,
        "max_iter": 5000,
        "limit": 200,
        "phi_range": (-1, 1),
        "f_target": F_TARGET_RASTRIGIN,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rosenbrock",
        "func":     rosenbrock_func,
        "dimension": DIMENSION,
        "bounds":   ROSENBROCK_BOUNDS,
        "colony_size": 20,
        "max_iter": 5000,
        "limit": 200,
        "phi_range": (-0.01, 0.01),
        "f_target": F_TARGET_ROSENBROCK,
        "interval_evals": INTERVAL_EVALS,    
    },
]

FUNCTIONS_FOR_CUCKOOSEARCH = [
    {
        "name":     "sphere",
        "func":     sphere,
        "dimension": DIMENSION,
        "bounds":   SPHERE_BOUNDS,
        "n_nests": 20,
        "max_iter": 4000,
        "pa": 0.25,
        "alpha": 0.01,
        "f_target": F_TARGET_SPHERE,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rastrigin",
        "func":     rastrigin,
        "dimension": DIMENSION,
        "bounds":   RASTRIGIN_BOUNDS,
        "n_nests": 20,
        "max_iter": 4000,
        "pa": 0.25,
        "alpha": 1,
        "f_target": F_TARGET_RASTRIGIN,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rosenbrock",
        "func":     rosenbrock_func,
        "dimension": DIMENSION,
        "bounds":   ROSENBROCK_BOUNDS,
        "n_nests": 30,
        "max_iter": 2668,
        "pa": 0.25,
        "alpha": 0.1,
        "f_target": F_TARGET_ROSENBROCK,
        "interval_evals": INTERVAL_EVALS,
    },
]

def suggest_gamma(bounds, fraction=0.5):
    """
    Sets gamma so fireflies can 'see' each other at fraction * domain_diagonal.
    fraction=1.0 → only interact when very close
    fraction=0.1 → interact across most of the space (more global)
    """
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    diagonal = np.sqrt(np.sum((upper - lower)**2))   # full diagonal of search space (maximum distance)
    char_dist = fraction * diagonal
    return 1.0 / (char_dist**2)
    # beta = beta0 * np.exp(-gamma * r**2) 
    # gamma =(fraction * maxD)^-2 
    # beta = beta0 / (e^(D / a * maxD)^2)
    # D/maxD ~ 0 - 1
    # beta ~ beta * 1/e^(fraction)^2 

FUNCTIONS_FOR_FIREFLY = [
    {
        "name":     "sphere",
        "func":     sphere,
        "dimension": DIMENSION,
        "bounds":   SPHERE_BOUNDS,
        "num_fireflies": 20,
        "max_iter": 520,
        "alpha": 0.01,
        "beta0": 1.0,
        "gamma": suggest_gamma(SPHERE_BOUNDS, fraction = 0.25),
        "f_target": F_TARGET_SPHERE,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rastrigin",
        "func":     rastrigin,
        "dimension": DIMENSION,
        "bounds":   RASTRIGIN_BOUNDS,
        "num_fireflies": 60,
        "max_iter": 55,
        "alpha": 0.2,
        "beta0": 0.5,
        "gamma": suggest_gamma(RASTRIGIN_BOUNDS, fraction = 0.25),
        "f_target": F_TARGET_RASTRIGIN,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rosenbrock",
        "func":     rosenbrock_func,
        "dimension": DIMENSION,
        "bounds":   ROSENBROCK_BOUNDS,
        "num_fireflies": 40,
        "max_iter": 125,
        "alpha": 0.1,
        "beta0": 0.5,
        "gamma": suggest_gamma(ROSENBROCK_BOUNDS, fraction = 0.25),
        "f_target": F_TARGET_ROSENBROCK,
        "interval_evals": INTERVAL_EVALS,
    },
]

FUNCTIONS_FOR_PSO = [
    {
        "name":     "sphere",
        "func":     sphere,
        "dimension": DIMENSION,
        "bounds":   SPHERE_BOUNDS,
        "num_particles": 20,
        "max_iter": 5000,
        "w": 0.7,
        "c1": 1.5,
        "c2": 1,
        "f_target": F_TARGET_SPHERE,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rastrigin",
        "func":     rastrigin,
        "dimension": DIMENSION,
        "bounds":   RASTRIGIN_BOUNDS,
        "num_particles": 60,
        "max_iter": 1666,
        "w": 0.7,
        "c1": 2,
        "c2": 0.5,
        "f_target": F_TARGET_RASTRIGIN,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rosenbrock",
        "func":     rosenbrock_func,
        "dimension": DIMENSION,
        "bounds":   ROSENBROCK_BOUNDS,
        "num_particles": 30,
        "max_iter": 3333,
        "w": 0.7,
        "c1": 1.5,
        "c2": 1.5,
        "f_target": F_TARGET_ROSENBROCK,
        "interval_evals": INTERVAL_EVALS,
    },
]

FUNCTIONS_FOR_DE = [
    {
        "name":     "sphere",
        "func":     sphere,
        "dimension": DIMENSION,
        "bounds":   SPHERE_BOUNDS,
        "pop_size": 80,
        "max_iter": 1250,
        "F": 0.7,
        "CR": 0.85,
        "f_target": F_TARGET_SPHERE,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rastrigin",
        "func":     rastrigin,
        "dimension": DIMENSION,
        "bounds":   RASTRIGIN_BOUNDS,
        "pop_size": 40,
        "max_iter": 2500,
        "F": 0.9,
        "CR": 0.1,
        "f_target": F_TARGET_RASTRIGIN,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rosenbrock",
        "func":     rosenbrock_func,
        "dimension": DIMENSION,
        "bounds":   ROSENBROCK_BOUNDS,
        "pop_size": 200,
        "max_iter": 499,
        "F": 0.45,
        "CR": 0.95,
        "f_target": F_TARGET_ROSENBROCK,
        "interval_evals": INTERVAL_EVALS,    
    },
]

FUNCTIONS_FOR_HILL_CLIMBING = [
    {
        "name":     "sphere",
        "func":     sphere,
        "dimension": DIMENSION,
        "bounds":   SPHERE_BOUNDS,
        "n_starts": 20,
        "max_iter": 5000,
        "step_size": 0.1,
        "f_target": F_TARGET_SPHERE,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rastrigin",
        "func":     rastrigin,
        "dimension": DIMENSION,
        "bounds":   RASTRIGIN_BOUNDS,
        "n_starts": 20,
        "max_iter": 5000,
        "step_size": 0.1,
        "f_target": F_TARGET_RASTRIGIN,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rosenbrock",
        "func":     rosenbrock_func,
        "dimension": DIMENSION,
        "bounds":   ROSENBROCK_BOUNDS,
        "n_starts": 20,
        "max_iter": 5000,
        "step_size": 0.1,
        "f_target": F_TARGET_ROSENBROCK,
        "interval_evals": INTERVAL_EVALS,    
    },
]

FUNCTIONS_FOR_SIMULATED_ANNEALING = [
    {
        "name":     "sphere",
        "func":     sphere,
        "dimension": DIMENSION,
        "bounds":   SPHERE_BOUNDS,
        "n_starts": 20,
        "max_iter": 5000,
        "step_size": 0.1,
        "initial_temp": 1000,
        "cooling_rate": 0.95,
        "f_target": F_TARGET_SPHERE,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rastrigin",
        "func":     rastrigin,
        "dimension": DIMENSION,
        "bounds":   RASTRIGIN_BOUNDS,
        "n_starts": 20,
        "max_iter": 5000,
        "step_size": 0.1,
        "initial_temp": 1000,
        "cooling_rate": 0.9,
        "f_target": F_TARGET_RASTRIGIN,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rosenbrock",
        "func":     rosenbrock_func,
        "dimension": DIMENSION,
        "bounds":   ROSENBROCK_BOUNDS,
        "n_starts": 20,
        "max_iter": 5000,
        "step_size": 0.1,
        "initial_temp": 1000,
        "cooling_rate": 0.95,
        "f_target": F_TARGET_ROSENBROCK,
        "interval_evals": INTERVAL_EVALS,    
    },
]

FUNCTIONS_FOR_GA = [
    {
        "name":     "sphere",
        "func":     sphere,
        "dimension": DIMENSION,
        "bounds":   SPHERE_BOUNDS,
        "pop_size": 200,
        "max_iter": 499,
        "crossover_rate": 0.5,
        "mutation_rate": 0.01,
        "mutation_scale_with_range": 0.05,
        "elite_ratio": 0.2,
        "f_target": F_TARGET_SPHERE,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rastrigin",
        "func":     rastrigin,
        "dimension": DIMENSION,
        "bounds":   RASTRIGIN_BOUNDS,
        "pop_size": 200,
        "max_iter": 499,
        "crossover_rate": 0.5,
        "mutation_rate": 0.01,
        "mutation_scale_with_range": 0.05,
        "elite_ratio": 0.2,
        "f_target": F_TARGET_RASTRIGIN,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rosenbrock",
        "func":     rosenbrock_func,
        "dimension": DIMENSION,
        "bounds":   ROSENBROCK_BOUNDS,
        "pop_size": 80,
        "max_iter": 1249,
        "crossover_rate": 0.5,
        "mutation_rate": 0.05,
        "mutation_scale_with_range": 0.05,
        "elite_ratio": 0.2,
        "f_target": F_TARGET_ROSENBROCK,
        "interval_evals": INTERVAL_EVALS,    
    },
]

FUNCTIONS_FOR_TLBO = [
    {
        "name":     "sphere",
        "func":     sphere,
        "dimension": DIMENSION,
        "bounds":   SPHERE_BOUNDS,
        "pop_size": 20,
        "max_iter": 2499,
        "f_target": F_TARGET_SPHERE,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rastrigin",
        "func":     rastrigin,
        "dimension": DIMENSION,
        "bounds":   RASTRIGIN_BOUNDS,
        "pop_size": 100,
        "max_iter": 499,
        "f_target": F_TARGET_RASTRIGIN,
        "interval_evals": INTERVAL_EVALS,
    },
    {
        "name":     "rosenbrock",
        "func":     rosenbrock_func,
        "dimension": DIMENSION,
        "bounds":   ROSENBROCK_BOUNDS,
        "pop_size": 80,
        "max_iter": 624,
        "f_target": F_TARGET_ROSENBROCK,
        "interval_evals": INTERVAL_EVALS,    
    },
]




# ABC_run_trials_multi_func_to_csv(FUNCTIONS_FOR_ABC=FUNCTIONS_FOR_ABC, folder_path=FOLDER_PATH, file_name="abc_results.csv", n_trials=30)

# PSO_run_trials_multi_func_to_csv(FUNCTIONS_FOR_PSO=FUNCTIONS_FOR_PSO, folder_path=FOLDER_PATH, file_name="pso_results.csv", n_trials=30)

# CuckooSearch_run_trials_multi_func_to_csv(FUNCTIONS_FOR_CUCKOOSEARCH=FUNCTIONS_FOR_CUCKOOSEARCH, folder_path=FOLDER_PATH, file_name="cuckoo_search_results.csv", n_trials=30)

# FireFly_run_trials_multi_func_to_csv(FUNCTIONS_FOR_FIREFLY=FUNCTIONS_FOR_FIREFLY, folder_path=FOLDER_PATH, file_name="firefly_results.csv", n_trials=30)

# DE_run_trials_multi_func_to_csv(FUNCTIONS_FOR_DE=FUNCTIONS_FOR_DE, folder_path=FOLDER_PATH, file_name="de_results.csv", n_trials=30)

# Hill_climbing_multi_starts_run_trials_multi_func_to_csv(FUNCTIONS_FOR_HILL_CLIMBING=FUNCTIONS_FOR_HILL_CLIMBING, folder_path=FOLDER_PATH, file_name="hill_climbing_results.csv", n_trials=30)

# Simulated_annealing_multi_starts_run_trials_multi_func_to_csv(FUNCTIONS_FOR_SIMULATED_ANNEALING=FUNCTIONS_FOR_SIMULATED_ANNEALING, folder_path=FOLDER_PATH, file_name="simulated_annealing_results.csv", n_trials=30)

TLBO_run_trials_multi_func_to_csv(FUNCTIONS_FOR_TLBO=FUNCTIONS_FOR_TLBO, folder_path=FOLDER_PATH, file_name="tlbo_results.csv", n_trials=30)

GA_run_trials_multi_func_to_csv(FUNCTIONS_FOR_GA=FUNCTIONS_FOR_GA, folder_path=FOLDER_PATH, file_name="ga_results.csv", n_trials=30)