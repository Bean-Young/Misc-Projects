% 任务 2: LU 分解法
% 实验人: 杨跃浙
A2 = [1 2 -1; 4 1 1; 2 3 2];
b2 = [4; 1; -4];

n = length(b2);
L = eye(n); % 初始化 L 为单位矩阵
U = A2;     % 初始化 U 为 A2

disp('初始矩阵 A:');
disp(A2);

% LU 分解
for k = 1:n-1
    for i = k+1:n
        L(i, k) = U(i, k) / U(k, k);
        U(i, k:n) = U(i, k:n) - L(i, k) * U(k, k:n);
    end
    % 输出当前 L 和 U
    disp(['第 ', num2str(k), ' 步 LU 分解后:']);
    disp('矩阵 L:');
    disp(L);
    disp('矩阵 U:');
    disp(U);
end

disp('LU 分解完成:');
disp('矩阵 L:');
disp(L);
disp('矩阵 U:');
disp(U);

% 解 Ly = b2 (前向替代)
y = zeros(n, 1);
disp('开始前向替代求解 Ly = b');
for i = 1:n
    y(i) = b2(i) - L(i, 1:i-1) * y(1:i-1);
    % 输出当前 y
    disp(['第 ', num2str(i), ' 步前向替代后，y:']);
    disp(y);
end

% 解 Ux = y (回代)
x2 = zeros(n, 1);
disp('开始回代求解 Ux = y');
for i = n:-1:1
    x2(i) = (y(i) - U(i, i+1:n) * x2(i+1:n)) / U(i, i);
    % 输出当前 x2
    disp(['第 ', num2str(i), ' 步回代后，x:']);
    disp(x2);
end

disp('LU 分解法最终解:');
disp(x2);