function [root, iterations] = newton_method_yyz(f, df, x0, tol, max_iter)
    x = x0;
    for k = 1:max_iter
        x_new = x - f(x) / df(x);
        if abs(x_new - x) < tol
            root = x_new;
            iterations = k;
            return;
        end
        x = x_new;
    end
    error('牛顿法未能在指定迭代次数内收敛');
end