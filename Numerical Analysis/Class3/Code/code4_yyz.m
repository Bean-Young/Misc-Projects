% 任务 4
% 实验人: 杨跃浙
% 已知数据点
x = [1, 2, 4, 5];
y = [1, 3, 4, 2];

% 边界条件 (自然边界条件 S''(1) = 0, S''(5) = 0)
% 使用 `spline` 函数支持的 clamped 方法
% 定义边界二阶导数
boundary_conditions = [0, 0]; % S''(1) = 0, S''(5) = 0

% 求解样条插值
pp = spline(x, [boundary_conditions(1), y, boundary_conditions(2)]); % piecewise polynomial

% 细化点绘制曲线
x_fine = linspace(min(x), max(x), 1000);
y_fine = ppval(pp, x_fine);

% 绘制结果
figure;
hold on;
plot(x, y, 'ro', 'MarkerSize', 8, 'DisplayName', 'Data Points');
plot(x_fine, y_fine, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Cubic Spline');
title('Cubic Spline Interpolation with Boundary Conditions');
xlabel('x');
ylabel('y');
legend('show');
grid on;
hold off;
