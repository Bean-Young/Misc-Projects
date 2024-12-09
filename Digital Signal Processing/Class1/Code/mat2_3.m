clc;clear;close all;
xn=[0,0,0,ones(1,99)];
hn=[0,0,0,3,0.5,0.2,0.7,0,-0.8];
y=conv(xn,hn);
stem(0:length(y)-1,y);
grid on;xlabel('n');ylabel('系统的响应');

clc; clear; close all; % 清空命令行窗口，清除工作区变量，关闭所有图形窗口

% 定义输入信号 xn，长度为 100，第一个值为 0，其余值为 1
xn = [0, ones(1, 99)];

% 定义系统的冲激响应 hn
hn = [0, 0, 0, 3, 0.5, 0.2, 0.7, 0, -0.8];

% 对输入信号 xn 和系统的冲激响应 hn 进行卷积运算
y = conv(xn, hn);

% 绘制卷积结果
stem(0:length(y)-1, y); % 使用 stem 函数绘制卷积结果 y
grid on; % 显示网格线
xlabel('n'); % 设置 x 轴标签
ylabel('系统的响应'); % 设置 y 轴标签
