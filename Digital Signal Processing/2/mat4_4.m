b=[1,3];
a=[1,3,2];
zplane(b,a);

% 清空所有变量，关闭所有图形窗口，清空命令行窗口
clear all; close all; clc;

% 定义滤波器系数
b = [1, 3]; % 分子系数
a = [1, 3, 2]; % 分母系数

% 绘制零极点图
zplane(b, a); % 绘制零极点图
title('Zero-Pole Plot'); % 设置图形标题
xlabel('Real Part'); % 设置 x 轴标签
ylabel('Imaginary Part'); % 设置 y 轴标签
grid on; % 显示网格线
