import numpy as np

def hill_climbing_algorithm(
    objective_function,
    DIMENSION,
    BOUNDS,
    N_STARTS=20,
    MAX_ITERATIONS=1000,
    STEP_SIZE=0.1,
):

    LOWER_BOUND = np.array([b[0] for b in BOUNDS])
    UPPER_BOUND = np.array([b[1] for b in BOUNDS])
    if len(LOWER_BOUND) != DIMENSION or len(UPPER_BOUND) != DIMENSION:
        raise ValueError("Bounds length must match dimension")

    # ----------------------------------------------------------

    # Initialize random solution
    current_positions = np.random.uniform(LOWER_BOUND, UPPER_BOUND, (N_STARTS, DIMENSION))
    current_fitnesses = np.array([objective_function(pos) for pos in current_positions])

    best_position = current_positions[np.argmin(current_fitnesses)].copy()
    best_fitness = np.min(current_fitnesses)

    for _ in range(MAX_ITERATIONS):
        neighbors = current_positions + np.random.uniform(-STEP_SIZE, STEP_SIZE, (N_STARTS, DIMENSION))
        neighbors = np.clip(neighbors, LOWER_BOUND, UPPER_BOUND)
        neighbor_fitnesses = np.array([objective_function(nei) for nei in neighbors])

        # Move to better neighbors (greedy)
        better_mask = neighbor_fitnesses < current_fitnesses

        current_positions[better_mask] = neighbors[better_mask]
        current_fitnesses[better_mask] = neighbor_fitnesses[better_mask]

        # Update global best
        current_best_idx = np.argmin(current_fitnesses)
        if current_fitnesses[current_best_idx] < best_fitness:
            best_fitness = current_fitnesses[current_best_idx]
            best_position = current_positions[current_best_idx].copy()

    return best_position, best_fitness
        