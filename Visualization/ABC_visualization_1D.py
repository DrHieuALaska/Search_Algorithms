###################################################################################

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def visualize_abc_animated(
    objective_function,
    NUM_FOOD_SOURCES=20,
    MAX_ITERATIONS=100,
    LIMIT=20,
    LOWER_BOUND=-10,
    UPPER_BOUND=10
):

    food_sources = np.random.uniform(LOWER_BOUND, UPPER_BOUND, NUM_FOOD_SOURCES)
    fitness = objective_function(food_sources)
    trial_counter = np.zeros(NUM_FOOD_SOURCES)
    
    # Store history for animation
    food_sources_history = [food_sources.copy()]
    best_history = [fitness.min()]

    def calculate_probabilities(fitness):
        inv_fit = 1 / (1 + fitness - np.min(fitness))
        return inv_fit / np.sum(inv_fit)

    best_fitness = np.min(fitness)

    # Run the complete algorithm and store states
    for iteration in range(MAX_ITERATIONS):
        # Employed Bees
        for i in range(NUM_FOOD_SOURCES):
            k = np.random.choice([j for j in range(NUM_FOOD_SOURCES) if j != i])
            phi = np.random.uniform(-1, 1)

            candidate = food_sources[i] + phi * (food_sources[i] - food_sources[k])
            candidate = np.clip(candidate, LOWER_BOUND, UPPER_BOUND)
            candidate_fitness = objective_function(candidate)

            if candidate_fitness < fitness[i]:
                food_sources[i] = candidate
                fitness[i] = candidate_fitness
                trial_counter[i] = 0
            else:
                trial_counter[i] += 1

        # Onlooker Bees
        probabilities = calculate_probabilities(fitness)
        for _ in range(NUM_FOOD_SOURCES):
            i = np.random.choice(NUM_FOOD_SOURCES, p=probabilities)
            k = np.random.choice([j for j in range(NUM_FOOD_SOURCES) if j != i])
            phi = np.random.uniform(-1, 1)

            candidate = food_sources[i] + phi * (food_sources[i] - food_sources[k])
            candidate = np.clip(candidate, LOWER_BOUND, UPPER_BOUND)
            candidate_fitness = objective_function(candidate)

            if candidate_fitness < fitness[i]:
                food_sources[i] = candidate
                fitness[i] = candidate_fitness
                trial_counter[i] = 0
            else:
                trial_counter[i] += 1

        # Scout Bees
        for i in range(NUM_FOOD_SOURCES):
            if trial_counter[i] >= LIMIT:
                food_sources[i] = np.random.uniform(LOWER_BOUND, UPPER_BOUND)
                fitness[i] = objective_function(food_sources[i])
                trial_counter[i] = 0

        current_best = np.min(fitness)
        if current_best < best_fitness:
            best_fitness = current_best

        best_history.append(best_fitness)
        food_sources_history.append(food_sources.copy())

    # -----------------------------
    # Create Animation
    # -----------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Prepare the objective function curve
    x_plot = np.linspace(LOWER_BOUND, UPPER_BOUND, 400)
    y_plot = objective_function(x_plot)

    def init():
        ax1.clear()
        ax2.clear()
        
        ax1.set_title("Convergence Curve")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Best Fitness")
        ax1.set_xlim(0, MAX_ITERATIONS)
        ax1.set_ylim(min(best_history) - 0.1, max(best_history) + 0.1)
        
        ax2.set_title("Bee Positions")
        ax2.set_xlabel("x")
        ax2.set_ylabel("f(x)")
        ax2.plot(x_plot, y_plot, 'b-', alpha=0.3)
        
        return []

    def update(frame):
        ax1.clear()
        ax2.clear()
        
        # Plot convergence curve
        ax1.plot(best_history[:frame+1], 'r-', linewidth=2)
        ax1.set_title(f"Convergence Curve (Iteration {frame}/{MAX_ITERATIONS})")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Best Fitness")
        ax1.set_xlim(0, MAX_ITERATIONS)
        ax1.set_ylim(min(best_history) - 0.1, max(best_history) + 0.1)
        ax1.grid(True, alpha=0.3)
        
        # Plot bee positions
        ax2.plot(x_plot, y_plot, 'b-', alpha=0.3, label='Objective Function')
        current_positions = food_sources_history[frame]
        ax2.scatter(current_positions, objective_function(current_positions), 
                   c='red', s=50, alpha=0.6, label='Bees')
        ax2.set_title(f"Bee Positions (Iteration {frame}/{MAX_ITERATIONS})")
        ax2.set_xlabel("x")
        ax2.set_ylabel("f(x)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        return []

    anim = FuncAnimation(fig, update, frames=MAX_ITERATIONS+1, 
                        init_func=init, blit=False, interval=50, repeat=False)
    
    plt.tight_layout()
    plt.show(block=True)
    
    return anim


def objective_function(x):
    return x**2 + 5 * np.cos(x)

# Run the animated visualization
anim = visualize_abc_animated(objective_function)