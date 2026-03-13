import numpy as np

def TLBO_algorithm(
    objective_function,
    DIMENSION,
    BOUNDS,
    POP_SIZE=50,
    MAX_ITER=500
):

    lower = np.array([b[0] for b in BOUNDS])
    upper = np.array([b[1] for b in BOUNDS])

    if lower.shape[0] != DIMENSION or upper.shape[0] != DIMENSION:
        raise ValueError("Bounds must match the specified dimension.")

    population = np.random.uniform(lower, upper, (POP_SIZE, DIMENSION))
    fitness = np.array([objective_function(ind) for ind in population])

    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_value = fitness[best_idx]

    for _ in range(MAX_ITER):

        # -----------------
        # Teacher Phase (student learns from the teacher)
        # -----------------
        teacher_idx = np.argmin(fitness)
        teacher = population[teacher_idx]

        mean = np.mean(population, axis=0)

        TF = np.random.randint(1, 3)      # Teaching factor (randomly 1 or 2)

        for i in range(POP_SIZE):

            r = np.random.rand(DIMENSION)

            new_solution = population[i] + r * (teacher - TF * mean)

            new_solution = np.clip(new_solution, lower, upper)

            new_fitness = objective_function(new_solution)

            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        # -----------------
        # Learner Phase (students learn from each other)
        # -----------------
        for i in range(POP_SIZE):

            j = np.random.randint(0, POP_SIZE)
            while j == i:
                j = np.random.randint(0, POP_SIZE)

            Xi = population[i]
            Xj = population[j]

            if fitness[i] < fitness[j]:
                new_solution = Xi + np.random.rand(DIMENSION) * (Xi - Xj)
            else:
                new_solution = Xi + np.random.rand(DIMENSION) * (Xj - Xi)

            new_solution = np.clip(new_solution, lower, upper)

            new_fitness = objective_function(new_solution)

            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_value:
            best_value = fitness[best_idx]
            best_solution = population[best_idx].copy()

    return best_solution, best_value