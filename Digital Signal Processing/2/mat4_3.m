clear all; close all; clc;
b=[1,0,-1];a=[1,0,-0.81];
figure(1);subplot(2,1,1);zplane(b,a);
h=impz(b,a);
subplot(2,1,2);stem(h);
title('系统单位冲激响应');
xlabel('n') ; ylabel('h(n)');
[H, W]= freqz(b,a);
figure(2);subplot(2,1,1);
plot(W/pi,abs(H));
title('幅度响应曲线');grid on;
xlabel('\omega x \pi'); ylabel('|H(e^j^\omega)|');
subplot(2,1,2);
plot(W/pi,angle(H));
title('相位响应曲线');
xlabel('\omega x\pi');ylabel('相角');grid on;

clear all; close all; clc; % 清空所有变量，关闭所有图形窗口，清空命令行窗口

% 定义滤波器系数
b = [1, 0, -1]; % 分子系数
a = [1, 0, -0.81]; % 分母系数

% 绘制零极点图
figure(1); % 创建第一个图形窗口
subplot(2,1,1); % 创建 2x1 子图的第一个子图
zplane(b, a); % 绘制零极点图

% 计算并绘制单位冲激响应
h = impz(b, a); % 计算单位冲激响应
subplot(2,1,2); % 创建 2x1 子图的第二个子图
stem(h); % 绘制单位冲激响应
title('系统单位冲激响应'); % 设置子图标题
xlabel('n'); % 设置 x 轴标签
ylabel('h(n)'); % 设置 y 轴标签

% 计算频率响应
[H, W] = freqz(b, a); % 计算频率响应 H 和对应的频率 W

% 绘制幅度响应曲线
figure(2); % 创建第二个图形窗口
subplot(2,1,1); % 创建 2x1 子图的第一个子图
plot(W/pi, abs(H)); % 绘制幅度响应曲线
title('幅度响应曲线'); % 设置子图标题
grid on; % 显示网格线
xlabel('\omega \times \pi'); % 设置 x 轴标签
ylabel('|H(e^{j\omega})|'); % 设置 y 轴标签

% 绘制相位响应曲线
subplot(2,1,2); % 创建 2x1 子图的第二个子图
plot(W/pi, angle(H)); % 绘制相位响应曲线
title('相位响应曲线'); % 设置子图标题
xlabel('\omega \times \pi'); % 设置 x 轴标签
ylabel('相角'); % 设置 y 轴标签
grid on; % 显示网格线
