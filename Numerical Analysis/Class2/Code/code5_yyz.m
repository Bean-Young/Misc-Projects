% 任务 4: Gauss-Seidel 法
% 实验人: 杨跃浙
A4 = [10 -1 -2; -1 10 -2; -1 -1 5];
b4 = [7.2; 8.3; 4.0];
x4 = zeros(size(b4)); % 初始值
tolerance = 1e-6; % 收敛容限
maxIter = 100; % 最大迭代次数
n = length(b4);

disp('初始解向量 x:');
disp(x4);

for k = 1:maxIter
    x_old = x4; % 保存旧解，用于计算误差
    disp(['第 ', num2str(k), ' 次迭代:']);
    for i = 1:n
        sum1 = 0;
        sum2 = 0;
        for j = 1:i-1
            sum1 = sum1 + A4(i, j) * x4(j);
        end
        for j = i+1:n
            sum2 = sum2 + A4(i, j) * x_old(j);
        end
        x4(i) = (b4(i) - sum1 - sum2) / A4(i, i);
        % 输出当前的 x(i)
        disp(['x(', num2str(i), ') = ', num2str(x4(i))]);
    end
    
    % 计算当前误差并显示
    error = norm(x4 - x_old, inf);
    disp(['当前误差: ', num2str(error)]);
    
    % 检查收敛条件
    if error < tolerance
        disp('达到收敛条件，迭代结束。');
        break;
    end
end

disp('Gauss-Seidel 法最终解:');
disp(x4);