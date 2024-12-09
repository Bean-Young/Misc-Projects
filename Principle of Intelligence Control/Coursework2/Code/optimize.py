import numpy as np
import matplotlib.pyplot as plt
from WPO import wolf_pack_optimization
from DBO import dung_beetle_algorithm
from PSO import particle_swarm_optimization
from BEE import bee_algorithm
from PIO import pigeon_optimization
from fun import sphere,ackley,beale,booth,matyas,rastrigin,rosenbrock,griewank,schwefel,zakharov
import numpy as np
import matplotlib.pyplot as plt
import os
# 修改的主函数
def optimize_and_save_plots():
    # 定义优化算法
    algorithms = {
        "PSO": particle_swarm_optimization,
        "WOA": wolf_pack_optimization,
        "Bee": bee_algorithm,
        "Pigeon": pigeon_optimization,
        "DBA": dung_beetle_algorithm  # 添加屎壳郎优化算法
    }

    # 定义测试函数
    test_functions = {
        "Sphere": (sphere, [-20, 20], 0),
        "Ackley": (ackley, [-20, 20], 0),
        "Beale": (beale, [-20, 20], 0),
        "Booth": (booth, [-20, 20], 0),
        "Matyas": (matyas, [-20, 20], 0),
        "Rastrigin": (rastrigin, [-5.12, 5.12], 0),
        "Rosenbrock": (rosenbrock, [-5, 10], 0),
        "Griewank": (griewank, [-600, 600], 0),
        "Schwefel": (schwefel, [-500, 500], 0),
        "Zakharov": (zakharov, [-5, 10], 0)
    }

    # 参数设置
    num_particles = 50
    tolerance = 1e-3
    max_iter = 1000


    for func_name, (func, bounds, optimal_value) in test_functions.items():
        dim = 2 if func_name in ["Beale", "Booth", "Matyas"] else 5

        for alg_name, alg_func in algorithms.items():
            print(f"Running {alg_name} on {func_name}...")
            plt.figure(figsize=(10, 8))  # 每种函数与算法组合绘制一张图

            for run in range(10):  # 每种算法运行 10 次
                if alg_name == "PSO":
                    _, best_score, history = alg_func(func, dim, bounds, num_particles, 0.5, 1.5, 1.5, tolerance, optimal_value, max_iter)
                elif alg_name == "Bee":
                    _, best_score, history = alg_func(func, dim, bounds, num_particles, 5, tolerance, optimal_value, max_iter)
                elif alg_name == "Pigeon":
                    _, best_score, history = alg_func(func, dim, bounds, num_particles, 0.1, 1.0, 0.5, tolerance, optimal_value, max_iter)
                elif alg_name == "WOA":
                    _, best_score, history = alg_func(func, dim, bounds, num_particles, 2, tolerance, optimal_value, max_iter)
                elif alg_name == "DBA":
                    _, best_score, history = alg_func(func, dim, bounds, num_particles, max_iter, tolerance, optimal_value)
                else:
                    raise ValueError(f"Unknown algorithm: {alg_name}")

                # 绘制每次运行的结果
                plt.plot(history, label=f"Run {run+1}")

            plt.xlabel("Iterations")
            plt.ylabel("Best Score")
            plt.title(f"{alg_name} on {func_name}")
            plt.legend()
            plt.grid()
            filename = os.path.join('/Users/youngbean/Desktop/opti/image', f"{func_name}_{alg_name}.png")
            plt.savefig(filename)  # 保存图像
            plt.close()

# 执行优化并保存收敛图像
optimize_and_save_plots()