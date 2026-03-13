import numpy as np

def particle_swarm_optimization(
    objective_func,
    DIMENSION,
    BOUNDS,
    NUM_PARTICLES=30,
    MAX_ITERATIONS=100,
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

    lower = np.array([b[0] for b in BOUNDS])
    upper = np.array([b[1] for b in BOUNDS])

    if lower.shape[0] != DIMENSION or upper.shape[0] != DIMENSION:
        raise ValueError("Bounds must match the specified dimension.")

    # Initialize particles
    positions = np.random.uniform(lower, upper, (NUM_PARTICLES, DIMENSION))
    velocities = np.zeros((NUM_PARTICLES, DIMENSION))

    # Personal best
    pbest_positions = positions.copy()
    pbest_scores = np.array([objective_func(p) for p in positions])

    # Global best
    gbest_index = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_index].copy()
    gbest_score = pbest_scores[gbest_index]

    for _ in range(MAX_ITERATIONS):

        r1 = np.random.rand(NUM_PARTICLES, DIMENSION)
        r2 = np.random.rand(NUM_PARTICLES, DIMENSION)

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
