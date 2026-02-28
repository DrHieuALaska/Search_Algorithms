import numpy as np

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

    for _ in range(max_iter):

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

    return gbest_position, gbest_score

def sphere(X):
    return X[0]**2 + X[1]**2

def rosenbrock_func(X):
    return np.sum(100*(X[1:] - X[:-1]**2)**2 + (X[:-1] - 1)**2)

def rastrigin(X):
    A = 10
    return A * len(X) + np.sum(X**2 - A * np.cos(2 * np.pi * X))

DIMENSION = 10

SPHERE_BOUNDS = [[-5.12, 5.12]] * DIMENSION
ROSENBROCK_BOUNDS = [[-5, 10]] * DIMENSION
RASTRIGIN_BOUNDS = [[-5.12, 5.12]] * DIMENSION

best_pos, best_val = particle_swarm_optimization(
    objective_func=rastrigin,
    dimension=DIMENSION,
    bounds=RASTRIGIN_BOUNDS,
    num_particles=300,
    max_iter=200
)

print("Best position:", best_pos)
print("Best value:", best_val)