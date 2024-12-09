clc; clear; close all
x=[1:7,6:-1:1];n=-2:10;
[x11,n11]=sigshift(x,n,5);[x12,n12]=sigshift(x,n,-4);
[xl,nl]=sigadd(2 *x11,n11,-3*x12,n12);
subplot(2,1,1);stem(nl,xl,'.');grid on;
xlabel('n');ylabel('x_1(n)=2x(n-5)-3x(n+4)');
[x21,n21]=sigfold(x,n);[x21,n21]=sigshift(x21,n21,3);
[x22,n22]=sigshift(x,n,2);[x22,n22]=sigmult(x,n,x22,n22);
[x2,n2]=sigadd(x21,n21,x22,n22);
subplot(2,1,2);stem(n2,x2,'.');grid on;
xlabel('n');ylabel('x_2(n)=x(3-n)+x(n)x(n-2)');

clc; clear; close all % 清空命令行窗口，清除工作区变量，关闭所有图形窗口

% 定义信号 x 和其对应的时间序列 n
x = [1:7, 6:-1:1]; % 信号 x
n = -2:10; % 时间序列 n 从 -2 到 10

% 对信号 x 进行时间偏移
[x11, n11] = sigshift(x, n, 5); % 信号右移 5
[x12, n12] = sigshift(x, n, -4); % 信号左移 4

% 对两个偏移后的信号进行线性组合
[xl, nl] = sigadd(2 * x11, n11, -3 * x12, n12); % 组合信号 2*x(n-5) - 3*x(n+4)

% 绘制组合后的信号
subplot(2,1,1); % 创建 2x1 子图的第一个子图
stem(nl, xl, '.'); % 使用 stem 函数绘制组合信号 xl
grid on; % 显示网格线
xlabel('n'); % 设置 x 轴标签
ylabel('x_1(n)=2x(n-5)-3x(n+4)'); % 设置 y 轴标签

% 对信号 x 进行折叠和偏移
[x21, n21] = sigfold(x, n); % 信号折叠
[x21, n21] = sigshift(x21, n21, 3); % 折叠后的信号右移 3

% 对信号 x 进行偏移和相乘
[x22, n22] = sigshift(x, n, 2); % 信号右移 2
[x22, n22] = sigmult(x, n, x22, n22); % 信号与其右移 2 后的信号相乘

% 对两个处理后的信号进行相加
[x2, n2] = sigadd(x21, n21, x22, n22); % 组合信号 x(3-n) + x(n)x(n-2)

% 绘制组合后的信号
subplot(2,1,2); % 创建 2x1 子图的第二个子图
stem(n2, x2, '.'); % 使用 stem 函数绘制组合信号 x2
grid on; % 显示网格线
xlabel('n'); % 设置 x 轴标签
ylabel('x_2(n)=x(3-n)+x(n)x(n-2)'); % 设置 y 轴标签


function [y,n] = sigshift(x,m,n0)
% SIGSHIFT Shifts a signal x by n0
% [y,n] = sigshift(x,m,n0)
% y is the shifted signal
% n is the new time index

n = m + n0; % 新的时间序列
y = x; % 信号内容不变
end

function [y,n] = sigadd(x1,n1,x2,n2)
% SIGADD Adds two signals
% [y,n] = sigadd(x1,n1,x2,n2)
% y is the resulting signal
% n is the new time index

n = min(min(n1),min(n2)):max(max(n1),max(n2)); % 新的时间序列范围
y1 = zeros(1,length(n)); y2 = y1; % 初始化两个信号
y1(ismember(n,n1)) = x1; % 将 x1 映射到新的时间序列
y2(ismember(n,n2)) = x2; % 将 x2 映射到新的时间序列
y = y1 + y2; % 相加
end

function [y,n] = sigfold(x,n)
% SIGFOLD Folds a signal x
% [y,n] = sigfold(x,n)
% y is the folded signal
% n is the new time index

y = fliplr(x); % 信号内容翻转
n = -fliplr(n); % 时间序列翻转
end

function [y,n] = sigmult(x1,n1,x2,n2)
% SIGMULT Multiplies two signals
% [y,n] = sigmult(x1,n1,x2,n2)
% y is the resulting signal
% n is the new time index

n = min(min(n1),min(n2)):max(max(n1),max(n2)); % 新的时间序列范围
y1 = zeros(1,length(n)); y2 = y1; % 初始化两个信号
y1(ismember(n,n1)) = x1; % 将 x1 映射到新的时间序列
y2(ismember(n,n2)) = x2; % 将 x2 映射到新的时间序列
y = y1 .* y2; % 相乘
end

