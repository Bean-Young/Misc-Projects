% 任务 3: 追赶法
% 实验人: 杨跃浙
a = [0; 6; 9; 10]; % 下对角线元素
b = [9; 9; 7; 3];  % 主对角线元素
c = [6; 9; 9; 0];  % 上对角线元素
d = [1; 2; 2; 4];  % 右端向量

n = length(b);
P = zeros(n, 1); % 存储修改后的对角线
Q = zeros(n, 1); % 存储右端向量的变换

disp('初始数据:');
disp('下对角线元素 a:');
disp(a);
disp('主对角线元素 b:');
disp(b);
disp('上对角线元素 c:');
disp(c);
disp('右端向量 d:');
disp(d);

% 前向消元
P(1) = b(1);
Q(1) = d(1) / P(1);
disp('前向消元过程:');
disp(['P(1) = ', num2str(P(1))]);
disp(['Q(1) = ', num2str(Q(1))]);

for i = 2:n
    P(i) = b(i) - a(i) * c(i-1) / P(i-1);
    Q(i) = (d(i) - a(i) * Q(i-1)) / P(i);
    % 输出每一步计算的 P 和 Q
    disp(['P(', num2str(i), ') = ', num2str(P(i))]);
    disp(['Q(', num2str(i), ') = ', num2str(Q(i))]);
end

% 回代求解
x3 = zeros(n, 1);
x3(n) = Q(n);
disp('回代过程:');
disp(['x(', num2str(n), ') = ', num2str(x3(n))]);

for i = n-1:-1:1
    x3(i) = Q(i) - c(i) * x3(i+1) / P(i);
    % 输出每一步计算的 x
    disp(['x(', num2str(i), ') = ', num2str(x3(i))]);
end

disp('追赶法最终解:');
disp(x3);