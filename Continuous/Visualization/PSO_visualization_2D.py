import numpy as np
import matplotlib.pyplot as plt

def particle_swarm_optimization(
    objective_func,
    dimension,
    bounds,
    num_particles=30,
    max_iter=100,
    w=0.7,          # inertia
    c1=1.5,         # cognitive
    c2=1.5          # social
):
    """
    Particle Swarm Optimization (PSO)

    objective_func : function(X) -> array of fitness values
                     X shape = (n_particles, dim)
    bounds         : (lower, upper)
                     lower/upper = list or np.array of size dim
    """

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    if lower.shape[0] != dimension or upper.shape[0] != dimension:
        raise ValueError("Bounds must match the specified dimension.")

    # Initialize particles
    positions = np.random.uniform(lower, upper, (num_particles, dimension))
    velocities = np.zeros((num_particles, dimension ))

    # Personal best
    pbest_positions = positions.copy()
    pbest_scores = np.array([objective_func(p) for p in positions])

    # Global best
    gbest_index = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_index].copy()
    gbest_score = pbest_scores[gbest_index]

    convergence = []
    history = []

    for _ in range(max_iter):

        history.append(positions.copy())

        r1 = np.random.rand(num_particles, dimension)
        r2 = np.random.rand(num_particles, dimension)

        # Update velocity
        velocities = (
            w * velocities
            + c1 * r1 * (pbest_positions - positions)
            + c2 * r2 * (gbest_position - positions)
        )

        # Update position
        positions = positions + velocities
        positions = np.clip(positions, lower, upper)

        # Evaluate
        scores = np.array([objective_func(p) for p in positions])

        # Update personal best
        better_mask = scores < pbest_scores
        pbest_positions[better_mask] = positions[better_mask]
        pbest_scores[better_mask] = scores[better_mask]

        # Update global best
        best_particle = np.argmin(pbest_scores)
        if pbest_scores[best_particle] < gbest_score:
            gbest_score = pbest_scores[best_particle]
            gbest_position = pbest_positions[best_particle].copy()
        convergence.append(gbest_score)

    return gbest_position, gbest_score, convergence, history

def sphere(X):
    return X[0]**2 + X[1]**2

# ---------------- RUN ALGORITHM ----------------

bounds = [(-5.12, 5.12), (-5.12, 5.12)]

best_pos, best_val, convergence, history = particle_swarm_optimization(
    objective_func=sphere,
    dimension=2,
    bounds=bounds,
    num_particles=40,
    max_iter=200
)

print("Best position:", best_pos)
print("Best value:", best_val)

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
        Z[i, j] = sphere([X[i, j], Y[i, j]])

plt.figure()
plt.contourf(X, Y, Z, levels=50)
plt.colorbar()

# Plot final firefly positions
final_positions = history[-1]
plt.scatter(final_positions[:, 0], final_positions[:, 1])

plt.title("Final Positions")
plt.show()


# ---------------- OPTIONAL: SIMPLE ANIMATION ----------------

plt.figure()

for positions in history:
    plt.clf()
    plt.contourf(X, Y, Z, levels=50)
    plt.scatter(positions[:, 0], positions[:, 1])
    plt.pause(0.1)

plt.show()