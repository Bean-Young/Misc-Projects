import numpy as np
import time

def pigeon_optimization(func, dim, bounds, num_pigeons, alpha, gamma, beta, tolerance, optimal_value, max_iter, seed=None):
    # 动态随机性或可重复性设置
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(int(time.time() * 1000) % (2**32))

    pigeons = np.random.uniform(bounds[0], bounds[1], (num_pigeons, dim))  # 初始化鸽子群
    scores = np.array([func(p) for p in pigeons])  # 计算每只鸽子的初始分数
    best_pigeon = pigeons[np.argmin(scores)]  # 初始化全局最优鸽子
    best_score = np.min(scores)  # 初始化全局最优分数
    history = [best_score]  # 记录历史最优分数

    initial_alpha = alpha
    initial_gamma = gamma

    for iteration in range(max_iter):
        # 动态调整步长（alpha）和缩放因子（gamma）
        alpha = initial_alpha * (1 - iteration / max_iter)
        gamma = initial_gamma * (1 - iteration / max_iter)

        for i in range(num_pigeons):
            # 更新每只鸽子的位置（引入局部扰动项）
            step = np.random.uniform(-alpha, alpha, dim) + beta * (best_pigeon - pigeons[i])
            pigeons[i] += gamma * step

            # 边界反弹机制
            for j in range(dim):
                if pigeons[i][j] < bounds[0]:
                    pigeons[i][j] = bounds[0] + np.random.uniform(0, alpha)
                elif pigeons[i][j] > bounds[1]:
                    pigeons[i][j] = bounds[1] - np.random.uniform(0, alpha)

            # 计算新的分数并更新个体最优
            score = func(pigeons[i])
            if score < scores[i]:
                scores[i] = score

        # 更新全局最优解
        current_best_score = np.min(scores)
        if current_best_score < best_score:
            best_score = current_best_score
            best_pigeon = pigeons[np.argmin(scores)]

        # 记录当前迭代的全局最优分数
        history.append(best_score)

        # 检查收敛条件
        if np.abs(best_score - optimal_value) < tolerance:
            print(f"Pigeon Converged at iteration {iteration + 1} with best_score = {best_score}")
            break
    else:
        print(f"Pigeon Reached maximum iterations ({max_iter}). Best score = {best_score}")

    return best_pigeon, best_score, history