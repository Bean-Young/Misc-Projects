clear all;clc;close all;
nx=0:14;xn=sin(0.4*nx);
nh=0:19;hn=0.9.^nh;
N1=length(xn);N2=length(hn);
N=N1+N2-1;
xn=[xn zeros(1,N-N1)];
hn=[hn zeros(1,N-N2)];
yn=fftconv(xn,hn, N);
nn=0:N-1;
subplot(3,1,1);stem(nn,xn);title('序列 x(n)');
subplot(3,1,2);stem(nn,hn);title('序列 h(n)');
subplot(3,1,3);stem(nn,real(yn));title('序列x(n)与h(n)的线性卷积y(n)');
xlabel('n');






clear all; clc; close all;

% 定义输入信号和滤波器系数
nx = 0:14; % 输入信号的时间序列
xn = sin(0.4 * nx); % 输入信号 x(n)
nh = 0:19; % 滤波器系数的时间序列
hn = 0.9 .^ nh; % 滤波器 h(n)

% 获取信号长度
N1 = length(xn); % 输入信号的长度
N2 = length(hn); % 滤波器的长度
N = N1 + N2 - 1; % 线性卷积的长度

% 扩展信号长度以进行 FFT
xn = [xn zeros(1, N - N1)]; % 将输入信号扩展到长度 N
hn = [hn zeros(1, N - N2)]; % 将滤波器扩展到长度 N

% 计算线性卷积
yn = fftconv(xn, hn, N); % 调用自定义 fftconv 函数进行卷积
nn = 0:N-1; % 卷积结果的时间序列

% 绘制输入信号、滤波器和卷积结果
subplot(3,1,1); stem(nn, xn); title('序列 x(n)');
xlabel('n'); ylabel('x(n)'); % 设置坐标轴标签
subplot(3,1,2); stem(nn, hn); title('序列 h(n)');
xlabel('n'); ylabel('h(n)'); % 设置坐标轴标签
subplot(3,1,3); stem(nn, real(yn)); title('序列 x(n) 与 h(n) 的线性卷积 y(n)');
xlabel('n'); ylabel('y(n)'); % 设置坐标轴标签

% 自定义 FFT 卷积函数
function y = fftconv(x, h, N)
    % FFTCONV 计算信号 x 和 h 的卷积，结果长度为 N
    % 输入:
    %   x - 输入信号
    %   h - 滤波器
    %   N - 卷积结果的长度
    % 输出:
    %   y - 卷积结果

    % 计算信号和滤波器的 FFT
    X = fft(x, N); % 输入信号的 FFT
    H = fft(h, N); % 滤波器的 FFT
    
    % 计算卷积结果的 FFT
    Y = X .* H; % 频域相乘
    
    % 计算卷积结果的逆 FFT
    y = ifft(Y); % 逆 FFT 得到时域卷积结果
end
