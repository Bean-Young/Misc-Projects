x1=[1,1,1];x2=[1,2,3,0,0,0,4];
y2=circonvfre(x1,x2,10)
y2=abs(y2)

% 清空工作区变量，清空命令行窗口，关闭所有图形窗口
clear; clc; close all;

% 定义输入信号
x1 = [1, 1, 1]; % 信号 x1
x2 = [1, 2, 3, 0, 0, 0, 4]; % 信号 x2

% 计算循环卷积
y2 = circonvfre(x1, x2, 10); % 调用自定义 circonvfre 函数计算长度为 10 的循环卷积
y2 = abs(y2); % 取循环卷积结果的绝对值

% 显示结果
disp('循环卷积结果的绝对值:');
disp(y2);

% 自定义基于频域的循环卷积函数
function y = circonvfre(x1, x2, N)
    % CIRCONVFRE 计算信号 x1 和 x2 的基于频域的循环卷积，结果长度为 N
    % 输入:
    %   x1 - 第一个输入信号
    %   x2 - 第二个输入信号
    %   N  - 循环卷积结果的长度
    % 输出:
    %   y  - 循环卷积结果

    % 将输入信号扩展到长度为 N
    x1 = [x1, zeros(1, N - length(x1))];
    x2 = [x2, zeros(1, N - length(x2))];
    
    % 计算信号的离散傅里叶变换 (DFT)
    X1 = fft(x1, N);
    X2 = fft(x2, N);
    
    % 计算频域内的乘积
    Y = X1 .* X2;
    
    % 计算乘积的逆离散傅里叶变换 (IDFT)
    y = ifft(Y);
end
