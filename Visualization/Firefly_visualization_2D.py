import numpy as np
import matplotlib.pyplot as plt

# ---------------- OBJECTIVE FUNCTION ----------------
# Rastrigin (classic multimodal test function)

def objective(x):
    A = 10
    return A * 2 + (x[0]**2 - A * np.cos(2 * np.pi * x[0])) \
                 + (x[1]**2 - A * np.cos(2 * np.pi * x[1]))


# ---------------- FIREFLY ALGORITHM ----------------

def firefly_algorithm_2d(
    objective_func,
    bounds,
    n_fireflies=25,
    max_iter=80,
    alpha=0.25,
    beta0=1.0,
    gamma=0.5
):
    dim = 2

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    fireflies = lower + (upper - lower) * np.random.rand(n_fireflies, dim)

    def fitness(x):
        return -objective_func(x)  # Minimization

    brightness = np.array([fitness(f) for f in fireflies])

    convergence = []
    history = []

    for _ in range(max_iter):

        history.append(fireflies.copy())

        for i in range(n_fireflies):
            for j in range(n_fireflies):

                if brightness[j] > brightness[i]:

                    r = np.linalg.norm(fireflies[i] - fireflies[j]) 
                    beta = beta0 * np.exp(-gamma * r**2)

                    step = beta * (fireflies[j] - fireflies[i])
                    random_step = alpha * (np.random.rand(dim) - 0.5)

                    fireflies[i] += step + random_step
                    fireflies[i] = np.clip(fireflies[i], lower, upper)

                    brightness[i] = fitness(fireflies[i])

        best_idx = np.argmax(brightness)
        convergence.append(objective_func(fireflies[best_idx]))

    return fireflies[best_idx], convergence, history


# ---------------- RUN ALGORITHM ----------------

bounds = [(-5.12, 5.12), (-5.12, 5.12)]

best_solution, convergence, history = firefly_algorithm_2d(
    objective,
    bounds,
    n_fireflies=100,
    max_iter=100,
    alpha=0.25,
    beta0=1.0,
    gamma=0.5
)

print("Best solution:", best_solution)
print("Best fitness:", objective(best_solution))


# ---------------- PLOT 1: CONVERGENCE ----------------

plt.figure()
plt.plot(convergence)
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("Firefly Algorithm Convergence")
plt.show()


# ---------------- PLOT 2: 2D LANDSCAPE + FIREFLIES ----------------

# Create mesh grid
x = np.linspace(bounds[0][0], bounds[0][1], 200)
y = np.linspace(bounds[1][0], bounds[1][1], 200)
X, Y = np.meshgrid(x, y)

Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = objective([X[i, j], Y[i, j]])

plt.figure()
plt.contourf(X, Y, Z, levels=50)
plt.colorbar()

# Plot final firefly positions
final_positions = history[-1]
plt.scatter(final_positions[:, 0], final_positions[:, 1])

plt.title("Final Firefly Positions")
plt.show()


# ---------------- OPTIONAL: SIMPLE ANIMATION ----------------

plt.figure()

for positions in history:
    plt.clf()
    plt.contourf(X, Y, Z, levels=50)
    plt.scatter(positions[:, 0], positions[:, 1])
    plt.pause(0.1)

plt.show()