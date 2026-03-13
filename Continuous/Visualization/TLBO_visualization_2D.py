import numpy as np
import matplotlib.pyplot as plt


def TLBO_algorithm_2d(
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

    convergence = []
    history = []

    for _ in range(MAX_ITER):

        history.append(population.copy())

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

        convergence.append(best_value)

    return best_solution, best_value, convergence, history


# ---------------- OBJECTIVE FUNCTION ----------------

def sphere(x):
    return x[0]**2 + x[1]**2


# ---------------- RUN ALGORITHM ----------------

bounds = [(-5.12, 5.12), (-5.12, 5.12)]

best_sol, best_fit, convergence, history = TLBO_algorithm_2d(
    sphere,
    DIMENSION=2,
    BOUNDS=bounds,
    POP_SIZE=30,
    MAX_ITER=200,
)

print("Best solution:", best_sol)
print("Best fitness:", best_fit)


# ---------------- PLOT 1: CONVERGENCE ----------------

plt.figure()
plt.plot(convergence)
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("TLBO Algorithm Convergence")
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