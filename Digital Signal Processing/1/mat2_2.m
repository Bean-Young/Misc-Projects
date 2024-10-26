x=[2,10,6,-1,-2,4,2];nx=-3:3;
h=[3,4,1,-4,3,2];nh=-1:4;
yl=conv(x,h)
[y2,ny2]=conv_m(x,nx,h,nh)
[y3,ny3]=convolution(x,nx,h,nh)
























clc; clear; close all; % 清空命令行窗口，清除工作区变量，关闭所有图形窗口

% 定义输入信号 x 及其对应的时间序列 nx
x = [2, 10, 6, -1, -2, 4, 2]; 
nx = -3:3; 

% 定义系统的冲激响应 h 及其对应的时间序列 nh
h = [3, 4, 1, -4, 3, 2]; 
nh = -1:4; 

% 使用内置的 conv 函数进行卷积运算
yl = conv(x, h);

% 使用自定义函数 conv_m 进行卷积运算
[y2, ny2] = conv_m(x, nx, h, nh);

% 使用自定义函数 convolution 进行卷积运算
[y3, ny3] = convolution(x, nx, h, nh);

% 定义自定义卷积函数 conv_m
function [y, ny] = conv_m(x, nx, h, nh)
    ny = nx(1) + nh(1) : nx(end) + nh(end); % 计算卷积结果的时间序列
    y = conv(x, h); % 使用内置的 conv 函数进行卷积运算
end

% 定义自定义卷积函数 convolution
function [y, ny] = convolution(x, nx, h, nh)
    ny_begin = nx(1) + nh(1); % 卷积结果的起始时间
    ny_end = nx(end) + nh(end); % 卷积结果的结束时间
    ny = ny_begin:ny_end; % 卷积结果的时间序列
    
    % 初始化卷积结果
    y = zeros(1, length(ny));
    
    % 进行卷积运算
    for i = 1:length(x)
        for j = 1:length(h)
            y(i + j - 1) = y(i + j - 1) + x(i) * h(j);
        end
    end
end
