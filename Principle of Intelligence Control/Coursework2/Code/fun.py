import numpy as np
# 定义优化问题函数
# Sphere 函数
def sphere(X):
    return sum(x**2 for x in X)

# Ackley 函数
def ackley(X):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(X)
    sum1 = sum(x**2 for x in X)
    sum2 = sum(np.cos(c * x) for x in X)
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)

# Beale 函数 (2D)
def beale(X):
    x, y = X
    return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2

# Booth 函数 (2D)
def booth(X):
    x, y = X
    return (x + 2 * y - 7)**2 + (2 * x + y - 5)**2

# Matyas 函数 (2D)
def matyas(X):
    x, y = X
    return 0.26 * (x**2 + y**2) - 0.48 * x * y

def rastrigin(X):
    n = len(X)
    return 10 * n + sum(x**2 - 10 * np.cos(2 * np.pi * x) for x in X)

def rosenbrock(X):
    return sum(100 * (X[i+1] - X[i]**2)**2 + (1 - X[i])**2 for i in range(len(X) - 1))

def griewank(X):
    sum1 = sum(x**2 for x in X)
    prod1 = np.prod([np.cos(x / np.sqrt(i + 1)) for i, x in enumerate(X)])
    return 1 + sum1 / 4000 - prod1

def schwefel(X):
    return 418.9829 * len(X) - sum(x * np.sin(np.sqrt(abs(x))) for x in X)

def zakharov(X):
    sum1 = sum(x**2 for x in X)
    sum2 = sum(0.5 * (i + 1) * x for i, x in enumerate(X))
    return sum1 + sum2**2 + sum2**4