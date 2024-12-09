% 实验人：杨跃浙
% 函数定义
g = @(x) exp(-x);        % 迭代函数
f = @(x) exp(-x) - x;    % 原函数
x0 = 0.1;                % 初始猜测值
tol = 1e-6;              % 收敛容差
max_iter = 50;           % 最大迭代次数

% 不动点迭代
[root, iterations] = fixed_point_method_yyz(g, x0, tol, max_iter);

% 输出结果
fprintf('不动点迭代法求得零点为: %.6f，迭代次数: %d\n', root, iterations);

% 绘制函数图像和零点
x = linspace(-1, 1, 100); % 修改绘制范围，从 -1 到 1
plot(x, f(x), 'b', 'LineWidth', 1.5);
hold on;
plot(root, f(root), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on;
xlabel('x');
ylabel('f(x)');
title('不动点迭代法求解零点');
legend('f(x)', '零点');
yline(0, '--k'); % 添加 x 轴辅助线