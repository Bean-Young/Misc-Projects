% 任务 1: 列主元消去法
% 实验人: 杨跃浙
A = [1 2 3; 5 4 10; 3 -0.1 1];
b = [1; 0; 2];

n = length(b);

disp('初始矩阵 A 和向量 b:');
disp(A);
disp(b);

for k = 1:n-1
    % 找到列主元并交换
    [~, maxIndex] = max(abs(A(k:n, k)));
    maxIndex = maxIndex + k - 1;
    if maxIndex ~= k
        % 交换矩阵行
        A([k, maxIndex], :) = A([maxIndex, k], :);
        b([k, maxIndex]) = b([maxIndex, k]);
        disp(['第 ', num2str(k), ' 列主元交换:']);
        disp('矩阵 A:');
        disp(A);
        disp('向量 b:');
        disp(b);
    end
    
    % 消元过程
    for i = k+1:n
        factor = A(i, k) / A(k, k);
        A(i, k:n) = A(i, k:n) - factor * A(k, k:n);
        b(i) = b(i) - factor * b(k);
    end
    
    % 显示消元后的矩阵和向量
    disp(['第 ', num2str(k), ' 列消元后:']);
    disp('矩阵 A:');
    disp(A);
    disp('向量 b:');
    disp(b);
end

% 回代求解
x1 = zeros(n, 1);
for i = n:-1:1
    x1(i) = (b(i) - A(i, i+1:n) * x1(i+1:n)) / A(i, i);
    
    % 显示每一步回代结果
    disp(['回代第 ', num2str(i), ' 行后 x:']);
    disp(x1);
end

disp('列主元消去法最终解:');
disp(x1);