% 任务 1: 主元消去法
% 实验人: 杨跃浙
A = [1 2 3; 5 4 10; 3 -0.1 1]; % 系数矩阵
b = [1; 0; 2];                 % 常数向量

% 获取矩阵的大小
n = length(b);

% 显示初始矩阵和向量
disp('初始矩阵 A 和向量 b:');
disp(A);
disp(b);

% 高斯消元过程
for k = 1:n-1
    disp(['当前第 ', num2str(k), ' 列消元:']);
    for i = k+1:n
        % 计算消元因子
        factor = A(i, k) / A(k, k);
        % 更新矩阵 A 的第 i 行
        A(i, k:n) = A(i, k:n) - factor * A(k, k:n);
        % 更新向量 b 的第 i 行
        b(i) = b(i) - factor * b(k);
    end
    
    % 显示消元后的矩阵和向量
    disp(['消元后矩阵 A（第 ', num2str(k), ' 列完成）:']);
    disp(A);
    disp('更新后的向量 b:');
    disp(b);
end

% 回代求解过程
x = zeros(n, 1); % 初始化解向量
for i = n:-1:1
    x(i) = (b(i) - A(i, i+1:n) * x(i+1:n)) / A(i, i);
    % 显示当前回代结果
    disp(['回代第 ', num2str(i), ' 行后解 x:']);
    disp(x);
end

% 输出最终结果
disp('主元消去法最终解:');
disp(x);