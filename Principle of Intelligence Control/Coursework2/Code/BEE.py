import numpy as np
import time

def bee_algorithm(func, dim, bounds, num_bees, elite_bees, tolerance, optimal_value, max_iter, seed=None):
    if seed is not None:
        np.random.seed(seed)  # 固定随机性
    else:
        np.random.seed(int(time.time() * 1000) % (2**32))  # 动态随机性

    bees = np.random.uniform(bounds[0], bounds[1], (num_bees, dim))
    scores = np.array([func(b) for b in bees])
    best_bee = bees[np.argmin(scores)]
    best_score = np.min(scores)
    history = [best_score]
    initial_range = 0.1  # 初始探索范围

    for iteration in range(max_iter):
        exploration_range = initial_range * (1 - iteration / max_iter)  # 动态探索范围
        for i in range(num_bees):
            for _ in range(elite_bees):
                new_bee = bees[i] + np.random.uniform(-exploration_range, exploration_range, dim)
                # 边界反弹策略
                for j in range(dim):
                    if new_bee[j] < bounds[0]:
                        new_bee[j] = bounds[0] + (bounds[0] - new_bee[j])
                    elif new_bee[j] > bounds[1]:
                        new_bee[j] = bounds[1] - (new_bee[j] - bounds[1])
                score = func(new_bee)
                if score < scores[i]:
                    scores[i] = score
                    bees[i] = new_bee

        # 更新全局最优解
        current_best_score = np.min(scores)
        if current_best_score < best_score:
            best_score = current_best_score
            best_bee = bees[np.argmin(scores)]

        history.append(best_score)

        # 动态调整容差
        current_tolerance = tolerance * (1 - iteration / max_iter)
        if abs(best_score - optimal_value) < current_tolerance:
            print(f"Bee Converged at iteration {iteration + 1} with best_score = {best_score}")
            break
    else:
        print(f"Bee Reached maximum iterations ({max_iter}). Best score = {best_score}")

    return best_bee, best_score, history