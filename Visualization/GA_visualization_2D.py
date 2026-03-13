import numpy as np
import matplotlib.pyplot as plt


def genetic_algorithm_2d(
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

    population = np.random.uniform(lower, upper, (POP_SIZE, DIMENSION))
    fitness = np.array([objective_function(ind) for ind in population])

    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_value = fitness[best_idx]

    def tournament_selection():
        i, j = np.random.choice(POP_SIZE, 2, replace=False)
        return population[i].copy() if fitness[i] < fitness[j] else population[j].copy()

    elite_count = max(1, int(elite_ratio * POP_SIZE))

    convergence = []
    history = []

    for _ in range(MAX_ITERATIONS):

        history.append(population.copy())

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
            mutation = np.random.normal(0, mutation_scale_with_range, DIMENSION) * (upper - lower)
            child += mutation * mutation_mask
            child = np.clip(child, lower, upper)

            new_population.append(child)

        population = np.array(new_population)
        fitness = np.array([objective_function(ind) for ind in population])

        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_value:
            best_value = fitness[current_best_idx]
            best_solution = population[current_best_idx].copy()

        convergence.append(best_value)

    return best_solution, best_value, convergence, history


# ---------------- OBJECTIVE FUNCTION ----------------

def sphere(x):
    return x[0]**2 + x[1]**2


# ---------------- RUN ALGORITHM ----------------

bounds = [(-5.12, 5.12), (-5.12, 5.12)]

best_sol, best_fit, convergence, history = genetic_algorithm_2d(
    sphere,
    DIMENSION=2,
    BOUNDS=bounds,
    POP_SIZE=30,
    MAX_ITERATIONS=200,
)

print("Best solution:", best_sol)
print("Best fitness:", best_fit)


# ---------------- PLOT 1: CONVERGENCE ----------------

plt.figure()
plt.plot(convergence)
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("Genetic Algorithm Convergence")
plt.show()


# ---------------- PLOT 2: 2D LANDSCAPE + POPULATION ----------------

# Create mesh grid
x = np.linspace(bounds[0][0], bounds[0][1], 200)
y = np.linspace(bounds[1][0], bounds[1][1], 200)
X, Y = np.meshgrid(x, y)

Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = sphere([X[i, j], Y[i, j]])

plt.figure()
plt.contourf(X, Y, Z, levels=50)
plt.colorbar()

# Plot final population positions
final_positions = history[-1]
plt.scatter(final_positions[:, 0], final_positions[:, 1])

plt.title("Final Population Positions")
plt.show()


# ---------------- OPTIONAL: SIMPLE ANIMATION ----------------

plt.figure()

for positions in history:
    plt.clf()
    plt.contourf(X, Y, Z, levels=50)
    plt.scatter(positions[:, 0], positions[:, 1])
    plt.pause(0.1)

plt.show()