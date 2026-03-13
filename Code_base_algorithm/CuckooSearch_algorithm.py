import numpy as np
import math

def cuckoo_search(
    objective_func,
    DIMENSION,
    BOUNDS,
    N_NESTS=25,
    MAX_ITERATIONS=100,
    pa=0.25,
    alpha=0.01
):
    """
    Cuckoo Search Optimization

    Parameters
    ----------
    objective_func : callable
        Function to minimize. Input: (nests) shape (n, dim)
    BOUNDS : tuple (lower, upper)
        lower/upper are arrays or scalars
    N_NESTS : int
        Number of nests (population size)
    MAX_ITERATIONS : int
        Number of iterations
    pa : float
        Discovery probability (fraction of nests replaced)
    alpha : float
        Step size scaling for Lévy flight

    Returns
    -------
    best_solution : ndarray
    best_fitness : float
    convergence : list
    """

    # set up bounds
    lower = np.array([b[0] for b in BOUNDS])
    upper = np.array([b[1] for b in BOUNDS])

    if lower.shape[0] != DIMENSION or upper.shape[0] != DIMENSION:
        raise ValueError("Bounds must match the specified dimension.")

    # Initialize nests
    nests = lower + (upper - lower) * np.random.rand(N_NESTS, DIMENSION)
    fitness = np.array([objective_func(nest) for nest in nests])

    best_idx = np.argmin(fitness)
    best_solution = nests[best_idx].copy()
    best_fitness = fitness[best_idx]

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

    for _ in range(MAX_ITERATIONS):

        # Generate new solutions via Lévy flights
        steps = levy_flight((N_NESTS, DIMENSION))
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
        n_replace = int(pa * N_NESTS)
        worst_idx = sorted_idx[-n_replace:]

        random_nests = lower + (upper - lower) * np.random.rand(n_replace, DIMENSION)
        random_fitness = np.array([objective_func(nest) for nest in random_nests])

        nests[worst_idx] = random_nests
        fitness[worst_idx] = random_fitness

        # Update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_solution = nests[best_idx].copy()
            best_fitness = fitness[best_idx]

    return best_solution, best_fitness