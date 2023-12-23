import numpy as np
from numba import njit
from src.utils import (compute_value_of_q_table, custom_argmax,compute_greedy_route, load_data, route_distance)


@njit
def eps_greedy_update(
    Q_table: np.ndarray,
    distances: np.ndarray,
    mask: np.ndarray,
    route: np.ndarray,
    epsilon: float,
    gamma: float,
    lr: float,
    N: int,
    Method: str      # Q-learning: Method: Q, Sarsa-learning: Method: SARSA
):
    """Updates Q table using epsilon greedy.

    Args:
        Q_table (np.ndarray): Input Q table.
        distances (np.ndarray): Distance matrix describing the TSP instance.
        mask (np.ndarray): Boolean mask giving which cities to ignore (already visited).
        route (np.ndarray): Route container.
        epsilon (float): exploration parameter for epsilon greedy.
        gamma (float): weight for future reward.
        lr (float): learning rate for q updates.

    Returns:
        np.ndarray: Updated Q table.
    """
    mask[0] = False
    next_visit = 0
    reward = 0
    for i in range(1, N):
        # Iteration i : choosing ith city to visit
        possible = np.where(mask == True)[0]
        current = route[i - 1]
        if len(possible) == 1:
            next_visit = possible[0]
            reward = -distances[int(current), int(next_visit)]
            # Reward for finishing the route
            max_next = -distances[int(next_visit), int(route[0])]

        elif (Method == "Q"):      # Q-learning
            u = np.random.random()
            if u < epsilon:
                # random choice amongst possible
                next_visit = np.random.choice(possible)
            else:
                next_visit, _ = custom_argmax(Q_table, int(current), mask)
            # update mask and route
            mask[next_visit] = False
            route[i] = next_visit
            reward = -distances[int(current), int(next_visit)]
            # Get max Q from new state
            _, max_next = custom_argmax(Q_table, int(next_visit), mask)
        
        elif (Method == "SARSA"):        # SARSA-learning
            u = np.random.random()
            if u < epsilon: # random choice amongst possible
                next_visit = np.random.choice(possible)
            else:
                next_visit, _ = custom_argmax(Q_table, int(current), mask)
            
            # update mask and route
            mask[next_visit] = False
            route[i] = next_visit
            reward = -distances[int(current), int(next_visit)]
            
            u = np.random.random()       # Target policy is same as behaviour policy
            possible_2 = np.where(mask == True)[0]
            if u < epsilon: # random choice amongst possible
                max_next =  np.random.choice(possible_2)
            else:
                max_next, _ = custom_argmax(Q_table, int(next_visit), mask)
        # updating Q
        Q_table[int(current), int(next_visit)] = Q_table[
            int(current), int(next_visit)
        ] + lr * (reward + gamma * max_next - Q_table[int(current), int(next_visit)])

    return Q_table


@njit
def Q_Update(
    Q_table: np.ndarray,
    distances: np.ndarray,
    epsilon: float,
    gamma: float,
    lr: float,
    epochs: int = 100,
    method = str            # "Q" OR "SARSA"
):
    """Performs simple Q learning algorithm, epsilon greedy, to learn
        a solution to the TSP.

    Args:
        Q_table (np.ndarray): Initial Q table.
        distances (np.ndarray): Distance matrix describing the TSP instance.
        epsilon (float): exploration parameter for epsilon greedy.
        gamma (float): weight for future reward.
        lr (float): learning rate for q updates.
        epochs (int, optional): Number of iterations. Defaults to 100.

    Returns:
        np.ndarray: Q table obtained after training..
        list: contains greedy distances for each epoch.
    """
    N = Q_table.shape[0]
    CompQ_table = Q_table.copy()
    mask = np.array([True] * N)
    route = np.zeros((N,))
    cache_distance_best = np.zeros((epochs,))
    cache_distance_comp = np.zeros((epochs,))
    for ep in range(epochs):
        CompQ_table = eps_greedy_update(
            CompQ_table, distances, mask, route, epsilon, gamma, lr, N, method
        ) 
        # update Q table only if best found so far is improved
        greedy_cost = compute_value_of_q_table(Q_table, distances)
        greedy_cost_comp = compute_value_of_q_table(CompQ_table, distances)
        cache_distance_best[ep] = greedy_cost
        cache_distance_comp[ep] = greedy_cost_comp
        if greedy_cost_comp < greedy_cost:
            Q_table[:, :] = CompQ_table[:, :]
        # resetting route and mask for next episode
        route[:] = 0
        mask[:] = True
    return Q_table, cache_distance_best, cache_distance_comp



def RL_Methods_Analysis(epsilon, gamma, epochs, method, learning_rate):
    """Q Learning & SARSA Learning methods is ran on each benchmark instance with different learning rate & epsilon value
    Figures monitoring progress are saved in figures/
    """
    # Loading test instances
    data = load_data()
    # start = time.time()
    # Running QLearning on each instance
    # METHODS = ["Q-learning", SARSA"]
    for c in data:
        Q_table = np.zeros((c, c))
        Q_table, cache_distance_best, cache_distance_comp = Q_Update(
            Q_table,
            data[c][0],         # distance matrix
            epsilon=epsilon,
            gamma=gamma,
            lr=learning_rate,
            epochs=epochs,
            method= method
        )

        # # Saving evaluation figure
        # trace_progress(
        #     cache_distance_comp,
        #     data[c][1],
        #     f"{c}_Cities_Best_distance_{data[c][1]}_Agent_exploration",
        # )

        greedy_route = compute_greedy_route(Q_table)
        greedy_cost = route_distance(greedy_route, data[c][0])
        # res[c] = greedy_cost
        print("Done RL")
        return(cache_distance_best, data[c][1], f"{c}_Cities_Best_distance_{data[c][1]}_Best_solution_found" )

if __name__ == '__main__':
    # RL_Methods_Analysis(epsilon=0.1, gamma=0.9, learning_rate=0.5,epochs=4000, method="SARSA")
    RL_Methods_Analysis(epsilon=0.1, gamma=0.9, learning_rate=0.5,epochs=4000, method="Q")
