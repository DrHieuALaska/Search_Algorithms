import numpy as np
import math
import matplotlib.pyplot as plt

def cuckoo_search(
    objective_func,
    dimension,
    bounds,
    n_nests=25,
    max_iter=100,
    pa=0.25,
    alpha=0.01
):
    # set up bounds
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    if lower.shape[0] != dimension or upper.shape[0] != dimension:
        raise ValueError("Bounds must match the specified dimension.")

    # Initialize nests
    nests = lower + (upper - lower) * np.random.rand(n_nests, dimension)
    fitness = np.array([objective_func(nest) for nest in nests])

    best_idx = np.argmin(fitness)
    best_solution = nests[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence = []
    history = []

    # Lévy flight generator
    def levy_flight(size):
        beta = 1.5
        sigma = (
            math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        u = np.random.randn(*size) * sigma
        v = np.random.randn(*size)
        step = u / np.abs(v) ** (1 / beta)

        return step

    for _ in range(max_iter):

        history.append(nests.copy())

        # Generate new solutions via Lévy flights
        steps = levy_flight((n_nests, dimension))
        new_nests = nests + alpha * steps * (nests - best_solution)

        # Apply bounds
        new_nests = np.clip(new_nests, lower, upper)

        new_fitness = np.array([objective_func(nest) for nest in new_nests])

        # Greedy selection
        improved = new_fitness < fitness
        nests[improved] = new_nests[improved]
        fitness[improved] = new_fitness[improved]

        # Replace fraction of worst nests
        sorted_idx = np.argsort(fitness)  
        n_replace = int(pa * n_nests)
        worst_idx = sorted_idx[-n_replace:]

        random_nests = lower + (upper - lower) * np.random.rand(n_replace, dimension)
        random_fitness = np.array([objective_func(nest) for nest in random_nests])

        nests[worst_idx] = random_nests
        fitness[worst_idx] = random_fitness

        # Update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_solution = nests[best_idx].copy()
            best_fitness = fitness[best_idx]

        convergence.append(best_fitness)

    return best_solution, best_fitness, convergence, history


def sphere(x):
    return x[0]**2 + x[1]**2

# ---------------- RUN ALGORITHM ----------------


bounds = [(-5.12, 5.12), (-5.12, 5.12)]

best_sol, best_fit, convergence, history = cuckoo_search(
    sphere,
    dimension=2,
    bounds=bounds,
    n_nests=30,
    max_iter=200
)

print("Best solution:", best_sol)
print("Best fitness:", best_fit)

# ---------------- PLOT 1: CONVERGENCE ----------------

plt.figure()
plt.plot(convergence)
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("Cuckoo Search Algorithm Convergence")
plt.show()


# ---------------- PLOT 2: 2D LANDSCAPE + FIREFLIES ----------------

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

# Plot final firefly positions
final_positions = history[-1]
plt.scatter(final_positions[:, 0], final_positions[:, 1])

plt.title("Final nests Positions")
plt.show()


# ---------------- OPTIONAL: SIMPLE ANIMATION ----------------

plt.figure()

for positions in history:
    plt.clf()
    plt.contourf(X, Y, Z, levels=50)
    plt.scatter(positions[:, 0], positions[:, 1])
    plt.pause(0.1)

plt.show()