function [result1, result2, diff_table] = newton_yyz(xdata, ydata, x)
    n = length(xdata); % 数据点数量
    diff_table = zeros(n, n); % 差商表
    diff_table(:, 1) = ydata'; % 第一列为 ydata

    % 计算差商表
    for j = 2:n
        for i = 1:(n - j + 1)
            diff_table(i, j) = ...
                (diff_table(i + 1, j - 1) - diff_table(i, j - 1)) ...
                / (xdata(i + j - 1) - xdata(i));
        end
    end

    [~, idx] = sort(abs(xdata - x)); % 按距离 x 的绝对值排序
    x_selected = xdata(idx); % 最近点的 x 数据
    y_selected = ydata(idx); % 最近点的 y 数据
    x0 = x_selected(1); % 最近的 x0
    x1 = x_selected(2); % 最近的 x1
    f_x0 = y_selected(1); % 对应的 f(x0)
    delta1 = diff_table(idx(1), 2); % Δ^1(x0)

    % 一次插值公式
    result1 = f_x0 + delta1 * (x - x0);

    delta2 = diff_table(x1, 3); % Δ^2(x1)
    f_x1 = y_selected(2); % 对应的 f(x1)
    delta11 = diff_table(x1, 2); % Δ^1(x1)
    % 二次插值公式
    result2 = f_x1 + delta11 * (x - x1) + delta2 * (x - x0) * (x - x1);

    % 画图
    % 定义细化的 x 值用于绘制插值曲线
    x_fine = linspace(min(xdata), max(xdata), 100);
    y_fine1 = f_x0 + delta1 * (x_fine - x0); % 一次插值曲线
    y_fine2 = f_x1 + delta11 * (x_fine - x1) + delta2 * (x_fine - x0) .* (x_fine - x1); % 二次插值曲线

    figure;
    hold on;
    % 原始数据点
    plot(xdata, ydata, 'o', 'MarkerSize', 8, 'DisplayName', 'Data Points');
    % 一次插值曲线
    plot(x_fine, y_fine1, '--', 'LineWidth', 1.5, 'DisplayName', 'Linear Interpolation');
    % 二次插值曲线
    plot(x_fine, y_fine2, '-', 'LineWidth', 1.5, 'DisplayName', 'Quadratic Interpolation');
    % 标记输入点
    plot(x, result1, 'x', 'MarkerSize', 10, 'LineWidth', 1.5, 'DisplayName', 'Linear Result');
    plot(x, result2, '+', 'MarkerSize', 10, 'LineWidth', 1.5, 'DisplayName', 'Quadratic Result');
    legend('show');
    title('Newton Interpolation');
    xlabel('x');
    ylabel('y');
    grid on;
    hold off;
end