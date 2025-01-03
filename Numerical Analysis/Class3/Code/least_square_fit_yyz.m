function [coefficients, fitted_values] = least_square_fit_yyz(x, y)
    % 构造矩阵 X 和向量 Y
    X = [ones(length(x), 1), x(:), x(:).^2]; % [1, x, x^2]
    Y = y(:); % 转为列向量
    
    % 最小二乘解
    coefficients = (X' * X) \ (X' * Y);
    
    % 计算拟合值
    fitted_values = X * coefficients;
    
    % 构造拟合多项式表达式
    a0 = coefficients(1);
    a1 = coefficients(2);
    a2 = coefficients(3);
    fprintf('Fitted Quadratic Polynomial: φ(x) = %.4f + %.4f*x + %.4f*x^2\n', a0, a1, a2);
    
    % 用于绘图的拟合曲线
    x_fine = linspace(min(x), max(x), 100); % 细化点
    y_fine = a0 + a1 * x_fine + a2 * x_fine.^2;
    
    % 绘制原始数据点和拟合曲线
    figure;
    hold on;
    % 原始数据点
    plot(x, y, 'ro', 'MarkerSize', 8, 'DisplayName', 'Original Data');
    % 拟合曲线
    plot(x_fine, y_fine, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Fitted Curve');
    legend('show');
    title(sprintf('Fitted Curve: φ(x) = %.4f + %.4f*x + %.4f*x^2', a0, a1, a2));
    xlabel('x');
    ylabel('y');
    grid on;
    hold off;
end