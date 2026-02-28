import numpy as np

def firefly_algorithm(
    objective_function,
    DIMENSION,
    BOUNDS,
    NUM_FIREFLIES=25,
    MAX_ITERATIONS=100,
    alpha=0.2,
    beta0=1.0,
    gamma=1.0,
):
    """
    Firefly Algorithm (FA)

    Parameters
    ----------
    objective_func : function
        Function to optimize. Takes a vector x and returns scalar fitness.
    dim : int
        Number of dimensions.
    bounds : list of tuples
        [(lower, upper), ...] for each dimension.
    n_fireflies : int
        Population size.
    max_iter : int
        Number of iterations.
    alpha : float
        Randomness parameter.
    beta0 : float
        Base attractiveness.
    gamma : float
        Light absorption coefficient.
    maximize : bool
        True for maximization, False for minimization.

    Returns
    -------
    best_solution : ndarray
    best_fitness : float
    """

    # --- Initialize population ---
    lower = np.array([b[0] for b in BOUNDS])
    upper = np.array([b[1] for b in BOUNDS])

    if lower.shape[0] != DIMENSION or upper.shape[0] != DIMENSION:
        raise ValueError("Bounds must match the specified dimension.")

    # initialize fireflies uniform randomly within bounds
    fireflies = lower + (upper - lower) * np.random.rand(NUM_FIREFLIES, DIMENSION)


    brightness = np.array([objective_function(f) for f in fireflies])
    convergence = []

    # --- Main loop ---
    for _ in range(MAX_ITERATIONS):
        for i in range(NUM_FIREFLIES):
            for j in range(NUM_FIREFLIES):

                if brightness[j] < brightness[i]:

                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = beta0 * np.exp(-gamma * r**2) # attractiveness decreases with distance 

                    step = beta * (fireflies[j] - fireflies[i]) # move towards brighter firefly
                    random_step = alpha * (np.random.rand(DIMENSION) - 0.5) # random perturbation alptha * [-0.5, 0.5]

                    fireflies[i] += step + random_step

                    # Apply bounds
                    fireflies[i] = np.clip(fireflies[i], lower, upper)

                    brightness[i] = objective_function(fireflies[i])
        convergence.append(np.min(brightness))

    # --- Select best ---
    best_idx = np.argmin(brightness)
    best_solution = fireflies[best_idx]
    best_fitness = objective_function(best_solution)

    return best_solution, best_fitness, convergence

def sphere(X):
    return np.sum(X**2)

def rosenbrock_func(X):
    return np.sum(100*(X[1:] - X[:-1]**2)**2 + (X[:-1] - 1)**2)

def rastrigin(X):
    A = 10
    return A * len(X) + np.sum(X**2 - A * np.cos(2 * np.pi * X))

DIMENSION = 10

SPHERE_BOUNDS = [[-5.12, 5.12]] * DIMENSION
ROSENBROCK_BOUNDS = [[-5, 10]] * DIMENSION
RASTRIGIN_BOUNDS = [[-5.12, 5.12]] * DIMENSION

def suggest_gamma(bounds, fraction=0.5):
    """
    Sets gamma so fireflies can 'see' each other at fraction * domain_diagonal.
    fraction=1.0 → only interact when very close
    fraction=0.1 → interact across most of the space (more global)
    """
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    diagonal = np.sqrt(np.sum((upper - lower)**2))   # full diagonal of search space (maximum distance)
    char_dist = fraction * diagonal
    return 1.0 / (char_dist**2)

best_x, best_fx, _ = firefly_algorithm(
    objective_function=rosenbrock_func,
    DIMENSION=DIMENSION,
    BOUNDS=ROSENBROCK_BOUNDS,
    NUM_FIREFLIES=20,
    MAX_ITERATIONS=460,
    alpha=0.1, # tune alpha to control randomness (exploration)
    beta0=1, # tune beta0 to control base attractiveness (exploitation)
    gamma=suggest_gamma(ROSENBROCK_BOUNDS, 0.5) # tune gamma to control how quickly attractiveness decreases with distance
)

print("Best solution:", best_x)
print("Best fitness:", best_fx)
