import numpy as np

def differential_evolution_with_evals(
    objective_function,
    DIMENSION,
    BOUNDS,
    POPULATION_SIZE=50,
    MAX_ITERATIONS=200,
    F=0.7,              # scaling factor
    CR=0.9,             # crossover rate
):
        
    # -------------------------
    # Bounds
    # -------------------------
    LOWER_BOUND = np.array([b[0] for b in BOUNDS])
    UPPER_BOUND = np.array([b[1] for b in BOUNDS])

    if LOWER_BOUND.shape[0] != DIMENSION or UPPER_BOUND.shape[0] != DIMENSION:
        raise ValueError("Bounds must match the specified dimension.")

    # -------------------------
    # Initialize population
    # -------------------------
    population = LOWER_BOUND + (UPPER_BOUND - LOWER_BOUND) * np.random.rand(POPULATION_SIZE, DIMENSION)
    fitness = np.array([objective_function(x) for x in population])

    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    # -------------------------
    # Main loop
    # -------------------------
    for _ in range(MAX_ITERATIONS):

        for i in range(POPULATION_SIZE):

            # --- Mutation (DE/rand/1) ---
            idxs = list(range(POPULATION_SIZE))
            idxs.remove(i)
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

            mutant = population[r1] + F * (population[r2] - population[r3])

            # Bound handling
            mutant = np.clip(mutant, LOWER_BOUND, UPPER_BOUND)

            # --- Crossover (binomial) ---
            trial = population[i].copy()
            j_rand = np.random.randint(0, DIMENSION)  # ensure at least one dimension from mutant

            for j in range(DIMENSION):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]

            # --- Selection (greedy) ---
            trial_fitness = objective_function(trial)

            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best_fitness:
                    best_solution = trial.copy()
                    best_fitness = trial_fitness

    return best_solution, best_fitness
