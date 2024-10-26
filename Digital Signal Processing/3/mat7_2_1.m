x1=[1,1,1];x2=[1,2,3,0,0,0,4];
y1=circonvtim(x1,x2,7)

% 清空工作区变量，清空命令行窗口，关闭所有图形窗口
clear; clc; close all;

% 定义输入信号
x1 = [1, 1, 1]; % 信号 x1
x2 = [1, 2, 3, 0, 0, 0, 4]; % 信号 x2

% 计算循环卷积
y1 = circonvtim(x1, x2, 7); % 调用自定义 circonvtim 函数计算长度为 7 的循环卷积

% 显示结果
disp('循环卷积结果:');
disp(y1);

% 自定义循环卷积函数
function y = circonvtim(x1, x2, N)
    % CIRCOVTIM 计算信号 x1 和 x2 的循环卷积，结果长度为 N
    % 输入:
    %   x1 - 第一个输入信号
    %   x2 - 第二个输入信号
    %   N  - 循环卷积结果的长度
    % 输出:
    %   y  - 循环卷积结果

    % 将输入信号扩展到长度为 N
    x1 = [x1, zeros(1, N - length(x1))];
    x2 = [x2, zeros(1, N - length(x2))];
    
    % 计算循环卷积
    y = zeros(1, N); % 初始化结果
    for n = 1:N
        for k = 1:N
            y(n) = y(n) + x1(k) * x2(mod(n-k, N) + 1);
        end
    end
end
