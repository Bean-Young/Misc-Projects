import numpy as np
# 狼群优化算法
def wolf_pack_optimization(func, dim, bounds, num_wolves, a_decay, tolerance, optimal_value, max_iter):
    # 初始化狼群位置
    wolves = np.random.uniform(bounds[0], bounds[1], (num_wolves, dim))
    scores = np.array([func(w) for w in wolves])
    
    # 排序狼群 (alpha, beta, delta 为狼群的前三名)
    sorted_indices = np.argsort(scores)
    alpha, beta, delta = wolves[sorted_indices[:3]]
    alpha_score, beta_score, delta_score = scores[sorted_indices[:3]]

    history = [alpha_score]

    for iteration in range(max_iter):
        a = 2 - iteration * (2 / max_iter)  # 动态调整范围因子 a

        for i in range(num_wolves):
            # 计算围攻和随机移动
            r1, r2 = np.random.rand(), np.random.rand()
            A1, C1 = 2 * a * r1 - a, 2 * r2
            D_alpha = abs(C1 * alpha - wolves[i])
            X1 = alpha - A1 * D_alpha

            r1, r2 = np.random.rand(), np.random.rand()
            A2, C2 = 2 * a * r1 - a, 2 * r2
            D_beta = abs(C2 * beta - wolves[i])
            X2 = beta - A2 * D_beta

            r1, r2 = np.random.rand(), np.random.rand()
            A3, C3 = 2 * a * r1 - a, 2 * r2
            D_delta = abs(C3 * delta - wolves[i])
            X3 = delta - A3 * D_delta

            # 更新狼的位置
            wolves[i] = (X1 + X2 + X3) / 3
            wolves[i] = np.clip(wolves[i], bounds[0], bounds[1])  # 确保位置在边界内

            # 计算新的得分
            score = func(wolves[i])
            if score < scores[i]:
                scores[i] = score

        # 更新 alpha, beta, delta
        sorted_indices = np.argsort(scores)
        alpha, beta, delta = wolves[sorted_indices[:3]]
        alpha_score, beta_score, delta_score = scores[sorted_indices[:3]]

        history.append(alpha_score)

        # 检查收敛条件
        if abs(alpha_score - optimal_value) < tolerance:
            print(f"WOA Converged at iteration {iteration + 1} with alpha_score = {alpha_score}")
            break

    else:
        print(f"WOA Reached maximum iterations ({max_iter}). Best score = {alpha_score}")

    return alpha, alpha_score, history
