import numpy as np
import matplotlib.pyplot as plt

def artificial_bee_colony(
    objective_function,
    DIMENSION,
    BOUNDS,
    COLONY_SIZE=40,
    MAX_ITERATIONS=200,
    LIMIT=20,
    PHI_RANGE=(-1, 1)
):
    """
    Artificial Bee Colony Optimization (Multi-dimensional)

    Parameters:
        objective_function : function f(x) where x is a vector
        DIMENSION          : number of variables
        LOWER_BOUND        : np.array of lower bounds 
        UPPER_BOUND        : np.array of upper bounds
        NUM_FOOD_SOURCES   : population size
        MAX_ITERATIONS     : number of iterations
        LIMIT              : abandonment limit

    Returns:
        best_solution (vector)
        best_fitness
    """

    # -----------------------------
    # Safty check for bounds
    # -----------------------------
    LOWER_BOUND = np.array([b[0] for b in BOUNDS])
    UPPER_BOUND = np.array([b[1] for b in BOUNDS])

    if LOWER_BOUND.shape[0] != DIMENSION or UPPER_BOUND.shape[0] != DIMENSION:
        raise ValueError("Bounds must be of the same dimension as specified by DIMENSION.")

    # -----------------------------
    # Initialization
    # -----------------------------
    NUM_FOOD_SOURCES = COLONY_SIZE // 2

    food_sources = np.random.uniform(
        LOWER_BOUND, UPPER_BOUND,
        (NUM_FOOD_SOURCES, DIMENSION)
    )

    fitness = np.array([objective_function(x) for x in food_sources])

    trial_counter = np.zeros(NUM_FOOD_SOURCES)

    best_solution = food_sources[np.argmin(fitness)].copy()
    best_fitness = np.min(fitness)
    convergence = []

    # -----------------------------
    # Probability Calculation
    # -----------------------------
    def calculate_probabilities(fitness):
        inv_fit = 1 / (1 + fitness - np.min(fitness))
        return inv_fit / np.sum(inv_fit)

    # -----------------------------
    # Main Loop
    # -----------------------------
    for _ in range(MAX_ITERATIONS):

        # =====================================
        # Employed Bee Phase
        # =====================================
        for i in range(NUM_FOOD_SOURCES):

            k = np.random.choice([j for j in range(NUM_FOOD_SOURCES) if j != i])
            phi = np.random.uniform(PHI_RANGE[0], PHI_RANGE[1], DIMENSION)

            candidate = food_sources[i] + phi * (food_sources[i] - food_sources[k])
            candidate = np.clip(candidate, LOWER_BOUND, UPPER_BOUND)

            candidate_fitness = objective_function(candidate)

            if candidate_fitness < fitness[i]:
                food_sources[i] = candidate
                fitness[i] = candidate_fitness
                trial_counter[i] = 0
            else:
                trial_counter[i] += 1

        # =====================================
        # Onlooker Bee Phase
        # =====================================
        probabilities = calculate_probabilities(fitness)

        for _ in range(NUM_FOOD_SOURCES):

            i = np.random.choice(NUM_FOOD_SOURCES, p=probabilities)
            k = np.random.choice([j for j in range(NUM_FOOD_SOURCES) if j != i])
            phi = np.random.uniform(PHI_RANGE[0], PHI_RANGE[1], DIMENSION)

            candidate = food_sources[i] + phi * (food_sources[i] - food_sources[k])
            candidate = np.clip(candidate, LOWER_BOUND, UPPER_BOUND)

            candidate_fitness = objective_function(candidate)

            if candidate_fitness < fitness[i]:
                food_sources[i] = candidate
                fitness[i] = candidate_fitness
                trial_counter[i] = 0
            else:
                trial_counter[i] += 1

        # =====================================
        # Scout Bee Phase
        # =====================================
        for i in range(NUM_FOOD_SOURCES):
            if trial_counter[i] >= LIMIT:
                food_sources[i] = np.random.uniform(LOWER_BOUND, UPPER_BOUND)
                fitness[i] = objective_function(food_sources[i])
                trial_counter[i] = 0
                # print(f"Scout bee reinitialized food source {i} due to trial limit.")

        # =====================================
        # Memorize Best Solution
        # =====================================
        current_best_index = np.argmin(fitness)
        current_best_fitness = fitness[current_best_index]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = food_sources[current_best_index].copy()
        
        convergence.append(best_fitness)

    return best_solution, best_fitness, convergence


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

best_x, best_fx, _ = artificial_bee_colony(
    objective_function=rosenbrock_func,
    DIMENSION=DIMENSION,
    BOUNDS=ROSENBROCK_BOUNDS,
    COLONY_SIZE=40,
    MAX_ITERATIONS=2000,
    LIMIT=200,
    PHI_RANGE=(-0.01, 0.01)
)

print("Best solution:", best_x)
print("Best fitness:", best_fx)
