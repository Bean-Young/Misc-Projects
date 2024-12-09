clear;clc;close all;
n=0:3;xn=[1,1,1,1];k1=0:1000;w=(pi/500)* k1;
X=xn*(exp(-j*pi/500)).^(n'*k1);magX=abs(X);angX=angle(X);N=length(xn);
nl=0:N-1;k=nl;nk=nl'*k;WN=exp(-j*2*pi/N);Wnk=WN.^nk;
Xk=xn*Wnk;magXk=abs(Xk);angXk=angle(Xk);
subplot(2,1,1);plot(w/pi,magX,'k--');hold on;stem(2*k/N,magXk);hold off; axis([0,2,0,5]); grid on;
xlabel('\omega(x\pi)');ylabel('幅度|X(k)|');
subplot(2,1,2);plot(w/pi,angX/pi,'k--');hold on;stem(2*k/N,angXk/pi);hold off;axis([0,2,-1,1]); grid on;
xlabel('\omega(x\pi)');ylabel('相位/\pi');

% 清空工作区变量，清空命令行窗口，关闭所有图形窗口
clear; clc; close all;

% 定义时间序列和信号
n = 0:3; % 时间序列 n 从 0 到 3
xn = [1, 1, 1, 1]; % 信号 xn

% 定义频率序列
k1 = 0:1000; % 频率索引 k1 从 0 到 1000
w = (pi/500) * k1; % 频率序列 w，范围从 0 到 2π

% 计算信号的离散时间傅里叶变换 (DTFT)
X = xn * (exp(-j * pi / 500)).^(n' * k1); % 计算 X(e^jω)
magX = abs(X); % 计算幅度
angX = angle(X); % 计算相位

% 计算信号的离散傅里叶变换 (DFT)
N = length(xn); % 信号长度
nl = 0:N-1; % 时间序列索引
k = nl; % 频率索引
nk = nl' * k; % 时间和频率索引的乘积
WN = exp(-j * 2 * pi / N); % DFT 矩阵元素
Wnk = WN .^ nk; % DFT 矩阵
Xk = xn * Wnk; % 计算 DFT X(k)
magXk = abs(Xk); % 计算 DFT 幅度
angXk = angle(Xk); % 计算 DFT 相位

% 绘制幅度响应曲线
subplot(2,1,1); % 创建 2x1 子图的第一个子图
plot(w/pi, magX, 'k--'); % 绘制 DTFT 幅度响应曲线，黑色虚线
hold on; % 保持当前图形
stem(2 * k / N, magXk); % 绘制 DFT 幅度响应，使用 stem 函数
hold off; % 结束保持当前图形
axis([0, 2, 0, 5]); % 设置坐标轴范围
grid on; % 显示网格线
xlabel('\omega (\times \pi)'); % 设置 x 轴标签
ylabel('幅度 |X(k)|'); % 设置 y 轴标签

% 绘制相位响应曲线
subplot(2,1,2); % 创建 2x1 子图的第二个子图
plot(w/pi, angX/pi, 'k--'); % 绘制 DTFT 相位响应曲线，黑色虚线
hold on; % 保持当前图形
stem(2 * k / N, angXk/pi); % 绘制 DFT 相位响应，使用 stem 函数
hold off; % 结束保持当前图形
axis([0, 2, -1, 1]); % 设置坐标轴范围
grid on; % 显示网格线
xlabel('\omega (\times \pi)'); % 设置 x 轴标签
ylabel('相位 / \pi'); % 设置 y 轴标签
