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

    # --- Select best ---
    best_idx = np.argmin(brightness)
    best_solution = fireflies[best_idx]
    best_fitness = objective_function(best_solution)

    return best_solution, best_fitness
