function [root, iterations] = secant_method_yyz(f, x0, x1, tol, max_iter)
    for k = 1:max_iter
        fx0 = f(x0);
        fx1 = f(x1);
        if abs(fx1 - fx0) < eps
            error('割线法失败：分母接近零');
        end
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0);
        if abs(x_new - x1) < tol
            root = x_new;
            iterations = k;
            return;
        end
        x0 = x1;
        x1 = x_new;
    end
    error('割线法未能在指定迭代次数内收敛');
end