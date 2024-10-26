clc;clear;close all
n=-10:20;
xl=3*sin(3*pi*n/5);
x2=cos(1.2*n);
subplot(2,1,1);stem(n,xl);
xlabel('n');ylabel('xl=3* sin(3\pin/5)');grid on;
subplot(2,1,2);stem(n,x2);
xlabel('n');ylabel('x2=cos(1.2\pin)');grid on;


















clc; clear; close all; % 清空命令行窗口，清除工作区变量，关闭所有图形窗口

n = -10:20; % 定义时间序列 n 从 -10 到 20

% 生成信号 xl，信号 xl 为 3*sin(3*pi*n/5)
xl = 3 * sin(3 * pi * n / 5);

% 生成信号 x2，信号 x2 为 cos(1.2*n)
x2 = cos(1.2 * n);

% 绘制信号 xl
subplot(2,1,1); % 创建 2x1 子图的第一个子图
stem(n, xl); % 使用 stem 函数绘制信号 xl
xlabel('n'); % 设置 x 轴标签
ylabel('xl=3*sin(3\pin/5)'); % 设置 y 轴标签
grid on; % 显示网格线

% 绘制信号 x2
subplot(2,1,2); % 创建 2x1 子图的第二个子图
stem(n, x2); % 使用 stem 函数绘制信号 x2
xlabel('n'); % 设置 x 轴标签
ylabel('x2=cos(1.2\pin)'); % 设置 y 轴标签
grid on; % 显示网格线
