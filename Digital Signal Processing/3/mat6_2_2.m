clear; clc; close all;
n = -5:5; xn = (-0.9).^n;
k1 = 0:1000; w = (pi/500) * k1;
X = xn * (exp(-j * pi / 500)) .^ (n' * k1);
magX = abs(X); angX = angle(X);
Xk = DFTmat(xn);
N = length(xn); k = 0:N-1;
Xk1 = Xk .* exp(j * 2 * pi * 5 * k / N);
magXk = abs(Xk1); angXk = angle(Xk1);
subplot(2,1,1); plot(w/pi, magX,'--'); hold on; stem(2 * k / N, magXk);
hold off; axis([0, 2, 0, 15]); grid on;
xlabel('\omega/\pi'); ylabel('|X(e^{j\omega})| 幅度');
subplot(2,1,2); plot(w/pi, angX/pi,'--'); hold on; stem(2 * k / N, angXk/pi);
hold off; axis([0, 2, -1, 1]); grid on;
xlabel('\omega/\pi'); ylabel('相位(\angle X(e^{j\omega}) / \pi)');
% 清空工作区变量，关闭所有图形窗口，清空命令行窗口
clear; close all; clc;

% 定义时间序列和信号
n = -5:5; % 时间序列 n 从 -5 到 5
x = (-0.9).^n; % 定义信号 x(n) = (-0.9)^n

% 定义频率序列
k = -200:200; % 频率索引 k 从 -200 到 200
w = (pi/100) * k; % 频率序列 w，范围从 -2π 到 2π

% 计算信号的离散时间傅里叶变换 (DTFT)
X = x * (exp(-j*pi/100)).^(n'*k); % 计算 X(e^jω)

% 计算幅度和相位
magX = abs(X); % 计算幅度
angX = angle(X); % 计算相位

% 绘制幅度响应曲线
subplot(2,1,1); % 创建 2x1 子图的第一个子图
plot(w/pi, magX); % 绘制幅度响应曲线
grid on; % 显示网格线
axis([-2, 2, 0, 15]); % 设置坐标轴范围
xlabel('\omega (\times \pi)'); % 设置 x 轴标签
ylabel('幅度 |H(e^{j\omega})|'); % 设置 y 轴标签

% 绘制相位响应曲线
subplot(2,1,2); % 创建 2x1 子图的第二个子图
plot(w/pi, angX/pi); % 绘制相位响应曲线
grid on; % 显示网格线
axis([-2, 2, -1, 1]); % 设置坐标轴范围
xlabel('\omega (\times \pi)'); % 设置 x 轴标签
ylabel('相位（弧度/\pi)'); % 设置 y 轴标签
