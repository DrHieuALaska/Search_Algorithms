import numpy as np
import os

def multi_start_simulated_multi_start_annealing_with_evals(
    objective_function,
    DIMENSION,
    BOUNDS,
    N_STARTS=10,
    MAX_ITERATIONS=5000,
    STEP_SIZE=0.1,
    INITIAL_TEMPERATURE=100,
    COOLING_RATE=0.995,
):

    LOWER_BOUND = np.array([b[0] for b in BOUNDS])
    UPPER_BOUND = np.array([b[1] for b in BOUNDS])
    if len(LOWER_BOUND) != DIMENSION or len(UPPER_BOUND) != DIMENSION:
        raise ValueError("Bounds length must match dimension")

   
    # Initialize random solution
    current_positions = np.random.uniform(LOWER_BOUND, UPPER_BOUND, (N_STARTS, DIMENSION))
    current_fitnesses = np.array([objective_function(pos) for pos in current_positions])

    temperature = INITIAL_TEMPERATURE
    best_position = current_positions[np.argmin(current_fitnesses)].copy()
    best_fitness = np.min(current_fitnesses)

    for _ in range(MAX_ITERATIONS):
        neighbors = current_positions + np.random.uniform(-STEP_SIZE, STEP_SIZE, (N_STARTS, DIMENSION))
        neighbors = np.clip(neighbors, LOWER_BOUND, UPPER_BOUND)
        neighbor_fitnesses = np.array([objective_function(nei) for nei in neighbors])

        # Move to better neighbors (greedy)
        better_mask = neighbor_fitnesses < current_fitnesses

        deltas = current_fitnesses - neighbor_fitnesses
        # Accept if better OR probabilistically if worse
        accept_mask = better_mask | (np.random.rand(N_STARTS) < np.exp(deltas / temperature))

        current_positions[accept_mask] = neighbors[accept_mask]
        current_fitnesses[accept_mask] = neighbor_fitnesses[accept_mask]

        # Update global best
        current_best_idx = np.argmin(current_fitnesses)
        if current_fitnesses[current_best_idx] < best_fitness:
            best_fitness = current_fitnesses[current_best_idx]
            best_position = current_positions[current_best_idx].copy()

        # Cool down
        temperature *= COOLING_RATE
        if temperature < 1e-10:
            break
    
    return best_position, best_fitness

        