clear;clc;close all
n=0:19;hn=0.5.^n;
m=0:101;xn=(1/3).^m;
y=conv(xn,hn);
stem(0:length(y)-1,y);
grid on;xlabel('n');
ylabel('x(n)和h(n)的卷积和计算结果');

clear; clc; close all; % 清空命令行窗口，清除工作区变量，关闭所有图形窗口

% 定义时间序列 n 和系统的冲激响应 hn
n = 0:19; % 时间序列 n 从 0 到 19
hn = 0.5 .^ n; % 系统的冲激响应 h(n) = 0.5^n

% 定义时间序列 m 和输入信号 xn
m = 0:101; % 时间序列 m 从 0 到 101
xn = (1/3) .^ m; % 输入信号 x(n) = (1/3)^m

% 对输入信号 xn 和系统的冲激响应 hn 进行卷积运算
y = conv(xn, hn);

% 绘制卷积结果
stem(0:length(y)-1, y); % 使用 stem 函数绘制卷积结果 y
grid on; % 显示网格线
xlabel('n'); % 设置 x 轴标签
ylabel('x(n)和h(n)的卷积和计算结果'); % 设置 y 轴标签
