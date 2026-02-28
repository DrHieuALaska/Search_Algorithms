import numpy as np
import math

def cuckoo_search(
    objective_func,
    dimension,
    bounds,
    n_nests=25,
    max_iter=100,
    pa=0.25,
    alpha=0.01
):
    """
    Cuckoo Search Optimization

    Parameters
    ----------
    objective_func : callable
        Function to minimize. Input: (nests) shape (n, dim)
    bounds : tuple (lower, upper)
        lower/upper are arrays or scalars
    n_nests : int
        Number of nests (population size)
    max_iter : int
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
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    if lower.shape[0] != dimension or upper.shape[0] != dimension:
        raise ValueError("Bounds must match the specified dimension.")

    # Initialize nests
    nests = lower + (upper - lower) * np.random.rand(n_nests, dimension)
    fitness = np.array([objective_func(nest) for nest in nests])

    best_idx = np.argmin(fitness)
    best_solution = nests[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence = []

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

    for _ in range(max_iter):

        # Generate new solutions via Lévy flights
        steps = levy_flight((n_nests, dimension))
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
        n_replace = int(pa * n_nests)
        worst_idx = sorted_idx[-n_replace:]

        random_nests = lower + (upper - lower) * np.random.rand(n_replace, dimension)
        random_fitness = np.array([objective_func(nest) for nest in random_nests])

        nests[worst_idx] = random_nests
        fitness[worst_idx] = random_fitness

        # Update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_solution = nests[best_idx].copy()
            best_fitness = fitness[best_idx]

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

best_sol, best_fit, history = cuckoo_search(
    rastrigin,
    dimension=DIMENSION,
    bounds=RASTRIGIN_BOUNDS,
    n_nests=40,
    max_iter=1600,
    pa=0.25,
    alpha=1
)

print("Best solution:", best_sol)
print("Best fitness:", best_fit)