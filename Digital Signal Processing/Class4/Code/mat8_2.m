clc; clear;
Xk = [36.0000, -4.0000 + 9.6569i,  -4.0000 + 4.0000i, -4.0000 + 1.6569i, -4.0000, -4.0000-1.6569i, -4.0000 - 4.0000i, -4.0000 - 9.6569i];
N = length(Xk); 
Xk1 = conj(Xk); 
xn1 = ditfft(Xk1);
xn1 = conj(xn1) / N;
xn1 = real(xn1)
xn2 = fft(Xk1); 
xn2 = conj(xn2) / N; 
xn2 = abs(xn2)
x3=ifft(Xk)

% 清空工作区变量，清空命令行窗口，关闭所有图形窗口
clear; clc; close all;

% 定义输入信号
xn = [1, 2, 3, 4, 5, 6, 7, 8]; % 信号 xn

% 调用自定义的 DIT-FFT 函数
Y_ditfft = ditfft(xn); % 使用自定义的 DIT-FFT 函数计算 FFT

% 调用 MATLAB 内置的 FFT 函数
Y_fft = fft(xn); % 使用 MATLAB 内置的 FFT 函数计算 FFT

% 显示结果
disp('自定义 DIT-FFT 结果:');
disp(Y_ditfft);
disp('MATLAB 内置 FFT 结果:');
disp(Y_fft);

% 自定义 DIT-FFT 函数
function X = ditfft(x)
    N = length(x); % 信号长度
    if N <= 1
        X = x;
    else
        % 拆分信号为偶数和奇数部分
        even = ditfft(x(1:2:end)); % 递归计算偶数部分的 FFT
        odd = ditfft(x(2:2:end)); % 递归计算奇数部分的 FFT
        W = exp(-2j * pi * (0:N/2-1) / N); % 计算旋转因子
        X = [even + W .* odd, even - W .* odd]; % 合并计算结果
    end
end
