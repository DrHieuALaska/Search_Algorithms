import numpy as np

def ant_colony_optimization(
    distance_matrix,
    n_ants=20,
    n_iterations=100,
    alpha=1.0,          # pheromone importance
    beta=2.0,           # distance importance
    evaporation=0.5,
    Q=100
):
    n_cities = len(distance_matrix)

    pheromone = np.ones((n_cities, n_cities))
    heuristic = 1 / (distance_matrix + 1e-10)

    best_path = None
    best_length = float("inf")

    for iteration in range(n_iterations):

        all_paths = []
        all_lengths = []

        for ant in range(n_ants):

            visited = []
            current_city = np.random.randint(n_cities)
            visited.append(current_city)

            while len(visited) < n_cities:

                probabilities = []
                for next_city in range(n_cities):

                    if next_city not in visited:
                        tau = pheromone[current_city][next_city] ** alpha
                        eta = heuristic[current_city][next_city] ** beta
                        probabilities.append((next_city, tau * eta))

                cities, probs = zip(*probabilities)
                probs = np.array(probs)
                probs /= probs.sum()

                next_city = np.random.choice(cities, p=probs)

                visited.append(next_city)
                current_city = next_city

            visited.append(visited[0])  # return to start

            length = sum(
                distance_matrix[visited[i]][visited[i+1]]
                for i in range(n_cities)
            )

            all_paths.append(visited)
            all_lengths.append(length)

            if length < best_length:
                best_length = length
                best_path = visited

        # Evaporation
        pheromone *= (1 - evaporation)

        # Deposit pheromone
        for path, length in zip(all_paths, all_lengths):
            deposit = Q / length
            for i in range(n_cities):
                a, b = path[i], path[i+1]
                pheromone[a][b] += deposit
                pheromone[b][a] += deposit

    return best_path, best_length


# Example random TSP
np.random.seed(0)

coords = np.random.rand(10, 2)

distance_matrix = np.sqrt(
    ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
)

best_path, best_length = ant_colony_optimization(distance_matrix)

print("Best path:", best_path)
print("Best length:", best_length)