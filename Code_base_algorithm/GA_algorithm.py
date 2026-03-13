import numpy as np

def genetic_algorithm(
    objective_function,
    DIMENSION,
    BOUNDS,
    POP_SIZE=50,
    MAX_ITERATIONS=500,
    crossover_rate=0.9,
    mutation_rate=0.1,
    mutation_scale_with_range=0.1,
    elite_ratio=0.05,
):

    # --- Initialize population ---
    lower = np.array([b[0] for b in BOUNDS])
    upper = np.array([b[1] for b in BOUNDS])

    if lower.shape[0] != DIMENSION or upper.shape[0] != DIMENSION:
        raise ValueError("Bounds must match the specified dimension.")

    # Initialize population
    population = np.random.uniform(lower, upper, (POP_SIZE, DIMENSION))
    fitness = np.array([objective_function(ind) for ind in population])

    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_value = fitness[best_idx]

    def tournament_selection():
        i, j = np.random.choice(POP_SIZE, 2, replace=False) # Ensure two distinct indices
        return population[i].copy() if fitness[i] < fitness[j] else population[j].copy()  
      
    # make sure at least one elite is preserved
    elite_count = max(1, int(elite_ratio * POP_SIZE)) 

    for _ in range(MAX_ITERATIONS):

        # --- Sort population by fitness ---
        elite_indices = np.argsort(fitness)[:elite_count]
        elites = population[elite_indices].copy()

        new_population = []

        # --- Preserve elites ---
        for e in elites:
            new_population.append(e)


        while len(new_population) < POP_SIZE:

            parent1 = tournament_selection()
            parent2 = tournament_selection()

            # Crossover
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand(DIMENSION)
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                child = parent1.copy()

            # Mutation
            mutation_mask = np.random.rand(DIMENSION) < mutation_rate
            mutation = np.random.normal(0, mutation_scale_with_range, DIMENSION) * (upper - lower) # Scale mutation by the range of bounds
            child += mutation * mutation_mask

            child = np.clip(child, lower, upper)

            new_population.append(child)

        population = np.array(new_population)
        fitness = np.array([objective_function(ind) for ind in population])

        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_value:
            best_value = fitness[current_best_idx]
            best_solution = population[current_best_idx].copy()

    return best_solution, best_value