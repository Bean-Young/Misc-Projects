% 任务 4
% 实验人: 杨跃浙
% 已知数据点
x = [2, 3, 5, 7, 8];
y = [1, 6, 22, 46, 61];

% 构造矩阵 X 和向量 Y
phi0 = ones(size(x));       % φ_0(x) = 1
phi1 = x.^2;                % φ_1(x) = x^2

% 构造法方程
A = [sum(phi0 .* phi0), sum(phi0 .* phi1); 
     sum(phi1 .* phi0), sum(phi1 .* phi1)];  % 矩阵 A
B = [sum(phi0 .* y); 
     sum(phi1 .* y)];                        % 矩阵 B

% 求解参数 a 和 b
coefficients = A \ B;  % [a; b]
a = coefficients(1);
b = coefficients(2);

% 显示结果
fprintf('Fitted Curve: φ(x) = %.2f + %.2f * x^2\n', a, b);

% 绘制拟合曲线
x_fine = linspace(min(x), max(x), 1000); % 细化点
y_fine = a + b * x_fine.^2;             % 拟合曲线

figure;
hold on;
plot(x, y, 'ro', 'MarkerSize', 8, 'DisplayName', 'Data Points');
plot(x_fine, y_fine, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Fitted Curve');
title('Least Squares Quadratic Fit');
xlabel('x');
ylabel('y');
legend('show');
grid on;
hold off;