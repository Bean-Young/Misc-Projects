function [root, iterations] = fixed_point_method_yyz(g, x0, tol, max_iter)
    x = x0;
    for k = 1:max_iter
        x_new = g(x);
        if abs(x_new - x) < tol
            root = x_new;
            iterations = k;
            return;
        end
        x = x_new;
    end
    error('不动点迭代法未能在指定迭代次数内收敛');
end