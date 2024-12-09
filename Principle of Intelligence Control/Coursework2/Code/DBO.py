import numpy as np

# 屎壳郎优化算法
def dung_beetle_algorithm(func, dim, bounds, num_agents, max_iter, tolerance, optimal_value):
    # 初始化位置和步长
    X = np.random.uniform(bounds[0], bounds[1], (num_agents, dim))
    scores = np.array([func(x) for x in X])
    best_agent = X[np.argmin(scores)]
    best_score = np.min(scores)
    history = [best_score]

    for iteration in range(max_iter):
        for i in range(num_agents):
            # 更新位置（模拟屎壳郎的导航行为）
            step_size = (bounds[1] - bounds[0]) * 0.1 * (1 - iteration / max_iter)  # 动态步长
            direction = np.random.uniform(-1, 1, dim)
            direction /= np.linalg.norm(direction)  # 归一化方向

            # 左右两点（屎壳郎感知的两个方向点）
            X_left = X[i] + step_size * direction
            X_right = X[i] - step_size * direction

            # 保证两点在边界内
            X_left = np.clip(X_left, bounds[0], bounds[1])
            X_right = np.clip(X_right, bounds[0], bounds[1])

            # 计算左右两点的目标函数值
            score_left = func(X_left)
            score_right = func(X_right)

            # 更新屎壳郎位置
            if score_left < score_right:
                X[i] = X_left
            else:
                X[i] = X_right

            # 保证屎壳郎位置在边界内
            X[i] = np.clip(X[i], bounds[0], bounds[1])

        # 更新全局最优解
        scores = np.array([func(x) for x in X])
        current_best_score = np.min(scores)
        if current_best_score < best_score:
            best_score = current_best_score
            best_agent = X[np.argmin(scores)]

        history.append(best_score)

        # 检查收敛条件
        if abs(best_score - optimal_value) < tolerance:
            print(f"Converged at iteration {iteration + 1} with best_score = {best_score}")
            break

    else:
        print(f"Reached maximum iterations ({max_iter}). Best score = {best_score}")

    return best_agent, best_score, history
