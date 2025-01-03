% 任务 2
% 实验人: 杨跃浙

f = @(x) 1 ./ (1 + x.^2);

% 定义区间
a = -5; % 区间左端点
b = 5;  % 区间右端点

% 用于绘图的细化点
x_fine = linspace(a, b, 1000);
y_fine = f(x_fine);

% 不同阶数的插值
n_values = [1, 2, 3, 4, 5, 6, 8, 10]; % 插值点个数
colors = lines(length(n_values)); % 不同阶数的颜色
figure;
hold on;

% 绘制原函数
plot(x_fine, y_fine, 'k-', 'LineWidth', 1.5, 'DisplayName', 'f(x) = 1/(1+x^2)');

% 遍历不同阶数
for k = 1:length(n_values)
    n = n_values(k);
    % 等距插值点
    x_nodes = linspace(a, b, n + 1);
    y_nodes = f(x_nodes);

    % 调用拉格朗日插值函数
    y_interp = lagrange_yyz(x_nodes, y_nodes, x_fine);

    % 绘制插值曲线
    plot(x_fine, y_interp, '-', 'LineWidth', 1.2, 'Color', colors(k, :), ...
        'DisplayName', sprintf('L_{%d}(x)', n));
end

% 图形美化
legend('show');
title('Runge Phenomenon');
xlabel('x');
ylabel('y');
