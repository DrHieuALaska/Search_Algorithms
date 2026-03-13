import numpy as np
import matplotlib.pyplot as plt

# ================= OBJECTIVE FUNCTION =================

def sphere(x):
    return x[0]**2 + x[1]**2

def objective(x):
    A = 10
    return A * 2 + (x[0]**2 - A * np.cos(2 * np.pi * x[0])) \
                 + (x[1]**2 - A * np.cos(2 * np.pi * x[1]))

# ================= ABC FUNCTION =================
def artificial_bee_colony_2d(
    objective_func,
    bounds,
    colony_size=30,
    max_iterations=100,
    limit=20
):
    dim = 2
    lower, upper = bounds

    food_number = colony_size // 2

    foods = np.random.uniform(lower, upper, (food_number, dim))
    fitness = np.array([objective_func(f) for f in foods])

    trial = np.zeros(food_number)

    best_index = np.argmin(fitness)
    best_solution = foods[best_index].copy()
    best_fitness = fitness[best_index]

    convergence = [best_fitness]

    # ================= VISUALIZATION SETUP =================
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Contour Plot ----
    x = np.linspace(lower, upper, 200)
    y = np.linspace(lower, upper, 200)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    contour = ax1.contourf(X, Y, Z, levels=50)
    plt.colorbar(contour, ax=ax1)

    scatter = ax1.scatter(foods[:, 0], foods[:, 1], c='red')
    best_point = ax1.scatter(best_solution[0], best_solution[1], c='blue', s=100)

    ax1.set_xlim(lower, upper)
    ax1.set_ylim(lower, upper)

    # ---- Convergence Plot ----
    line, = ax2.plot([], [])
    ax2.set_xlim(0, max_iterations)
    ax2.set_ylim(0, best_fitness * 1.1)
    ax2.set_title("Convergence")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best Fitness")

    # ================= MAIN LOOP =================
    for iteration in range(max_iterations):

        # ===== EMPLOYED BEES =====
        for i in range(food_number):
            k = np.random.randint(food_number)
            while k == i:
                k = np.random.randint(food_number)

            phi = np.random.uniform(-1, 1, dim)
            candidate = foods[i] + phi * (foods[i] - foods[k])
            candidate = np.clip(candidate, lower, upper)

            candidate_fitness = objective_func(candidate)

            if candidate_fitness < fitness[i]:
                foods[i] = candidate
                fitness[i] = candidate_fitness
                trial[i] = 0
            else:
                trial[i] += 1

        # ===== ONLOOKER BEES =====
        prob = 1 / (1 + fitness)
        prob /= prob.sum()

        for _ in range(food_number):
            i = np.random.choice(food_number, p=prob)

            k = np.random.randint(food_number)
            while k == i:
                k = np.random.randint(food_number)

            phi = np.random.uniform(-1, 1, dim)
            candidate = foods[i] + phi * (foods[i] - foods[k])
            candidate = np.clip(candidate, lower, upper)

            candidate_fitness = objective_func(candidate)

            if candidate_fitness < fitness[i]:
                foods[i] = candidate
                fitness[i] = candidate_fitness
                trial[i] = 0
            else:
                trial[i] += 1

        # ===== SCOUT BEES =====
        for i in range(food_number):
            if trial[i] > limit:
                foods[i] = np.random.uniform(lower, upper, dim)
                fitness[i] = objective_func(foods[i])
                trial[i] = 0

        # ===== BEST UPDATE =====
        best_index = np.argmin(fitness)
        if fitness[best_index] < best_fitness:
            best_solution = foods[best_index].copy()
            best_fitness = fitness[best_index]

        convergence.append(best_fitness)

        # ===== VISUAL UPDATE =====
        scatter.set_offsets(foods)
        best_point.set_offsets(best_solution)

        ax1.set_title(f"Iteration {iteration+1}")

        line.set_data(range(len(convergence)), convergence)
        ax2.set_ylim(min(convergence) * 0.9, max(convergence) * 1.1)

        plt.pause(0.1)

    plt.ioff()
    plt.show()

    return best_solution, best_fitness


# ================= RUN =================
best_solution, best_fitness = artificial_bee_colony_2d(
    objective_func=objective,
    bounds=(-5, 5),
    colony_size=40,
    max_iterations=100
)

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)