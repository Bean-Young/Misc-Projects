% 实验人：杨跃浙
% 函数定义
f = @(x) exp(x) - x - 5; % 原函数
x0 = 3.8;                % 初始猜测值 1
x1 = 3.6;                % 初始猜测值 2
tol = 1e-6;              % 收敛容差
max_iter = 50;           % 最大迭代次数

% 割线法迭代
[root, iterations] = secant_method_yyz(f, x0, x1, tol, max_iter);

% 输出结果
fprintf('割线法求得零点为: %.6f，迭代次数: %d\n', root, iterations);

% 绘制函数图像和零点
x = linspace(0, 5, 100); % 修改绘制范围，从 0 到 5
plot(x, f(x), 'b', 'LineWidth', 1.5);
hold on;
plot(root, f(root), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on;
xlabel('x');
ylabel('f(x)');
title('割线法求解零点');
legend('f(x)', '零点');
yline(0, '--k'); % 添加 x 轴辅助线