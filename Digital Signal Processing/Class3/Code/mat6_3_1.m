clear; clc; close all
N=1000;w=[0:N-1]*2*pi/N;
X=(sin(2*w)./sin(w/2)).*exp(-j*3*w/2);
magX=abs(X);angX=angle(X);
subplot(2,1,1);plot(w/pi,magX);grid on;
xlabel('\omegn(x\pi)');ylabel('幅度|X(e^j^\omega)|');
subplot(2,1,2);plot(w/pi,angX); grid on;
xlabel('\omegn(x\pi)');ylabel('相位');
% 清空工作区变量，清空命令行窗口，关闭所有图形窗口
clear; clc; close all;

% 定义频率序列
N = 1000; % 频率点数
w = [0:N-1] * 2 * pi / N; % 频率序列，从 0 到 2π

% 计算 X(e^jω)
X = (sin(2*w) ./ sin(w/2)) .* exp(-j * 3 * w / 2); % 计算信号的频谱

% 计算幅度和相位
magX = abs(X); % 计算幅度
angX = angle(X); % 计算相位

% 绘制幅度响应曲线
subplot(2,1,1); % 创建 2x1 子图的第一个子图
plot(w/pi, magX); % 绘制幅度响应曲线
grid on; % 显示网格线
xlabel('\omega (\times \pi)'); % 设置 x 轴标签
ylabel('幅度 |X(e^{j\omega})|'); % 设置 y 轴标签

% 绘制相位响应曲线
subplot(2,1,2); % 创建 2x1 子图的第二个子图
plot(w/pi, angX); % 绘制相位响应曲线
grid on; % 显示网格线
xlabel('\omega (\times \pi)'); % 设置 x 轴标签
ylabel('相位'); % 设置 y 轴标签
