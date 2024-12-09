import numpy as np
# 粒子群优化算法
def particle_swarm_optimization(func, dim, bounds, num_particles, w, c1, c2, tolerance, optimal_value, max_iter):
    # 初始化粒子位置和速度
    X = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
    V = np.random.uniform(-abs(bounds[1] - bounds[0]) * 0.1, abs(bounds[1] - bounds[0]) * 0.1, (num_particles, dim))
    personal_best_X = np.copy(X)
    personal_best_scores = np.array([func(x) for x in X])
    global_best_X = personal_best_X[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)
    history = [global_best_score]

    for iteration in range(max_iter):
        prev_best_score = global_best_score  # 记录上一次的最优值

        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            V[i] = w * V[i] + c1 * r1 * (personal_best_X[i] - X[i]) + c2 * r2 * (global_best_X - X[i])
            X[i] += V[i]
            X[i] = np.clip(X[i], bounds[0], bounds[1])  # 保证粒子位置在边界内

            score = func(X[i])
            if score < personal_best_scores[i]:
                personal_best_X[i] = X[i]
                personal_best_scores[i] = score

        global_best_X = personal_best_X[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        history.append(global_best_score)

        # 检查是否满足收敛条件
        if abs(global_best_score - optimal_value) < tolerance:
            print(f"Converged at iteration {iteration + 1} with global_best_score = {global_best_score}")
            break

    else:
        print(f"Reached maximum iterations ({max_iter}). Best score = {global_best_score}")

    return global_best_X, global_best_score, history
