clear;close all;clc
fs=10000;fs1=1000;fs2=400;fs3=200;
t=0:1/fs:0.1;
A=444.128;a=50*sqrt(2)*pi;b=a;
xa=exp(-a*t).*sin(b*t);
k=0:511;f=fs*k/512;
w=2*pi*k/512;
Xa=xa*exp(-j*[1:length(xa)]'*w);
T1=1/fs1;t1=0:T1:0.1;
x1=A*exp(-a.*t1).*sin(b*t1);
X1=x1*exp(-j*[1:length(x1)]'*w);

T2=1/fs2;t2=0:T2:0.1;
x2=A*exp(-a.*t2).*sin(b*t2);
X2=x2*exp(-j*[1:length(x2)]'*w);

T3=1/fs3;t3=0:T3:0.1;
x3=A*exp(-a.*t3).*sin(b*t3);
X3=x3*exp(-j*[1:length(x3)]'*w);

figure(1);
subplot(2,2,1);plot(t,xa);

axis([0,max(1),min(xa),max(xa)]);title('模拟信号');
xlabel('t(s)');ylabel('Xa(t)');line([0,max(t)],[0,0])
subplot(2,2,2);plot(f,abs(Xa)/max(abs(Xa)));
title('模拟信号的幅度频谱');axis([0,500,0,1])
xlabel('f(Hz)');ylabel('|Xa(jf)|');
subplot(2,2,3);stem(t1,x1,'.');
line([0,max(t1)],[0,0]);axis([0,max(t1),min(x1),max(x1)])
title('抽样序列x1(n)(fs1=1kHz)');xlabel('n');ylabel('X1(n)');
f1=fs1* k/512;
subplot(2,2,4);plot(f1,abs(X1)/max(abs(X1)));
title('x1(n)的幅度谱');xlabel('f(Hz)');ylabel('|X1(jf)|');
figure(2);
subplot(2,2,1);stem(t2,x2,'.');
line([0,max(t2)],[0,0]);axis([0,max(t2),min(x2),max(x2)]);
title('抽样序列 x2(n)(fs2=400Hz)');xlabel('n');ylabel('X2(n)');
f=fs2*k/512;
subplot(2,2,2);plot(f,abs(X2)/max(abs(X2)));
title('x2(n)的幅度谱');xlabel('f(Hz)');ylabel('|X2(jf)|');
subplot(2,2,3);stem(t3,x3,'.');
line([0,max(t3)],[0,0]);axis([0,max(t3),min(x3),max(x3)]);
title('抽样序列 x3(n)(fs3=200Hz)');xlabel('n');ylabel('X3(n)');
f=fs3*k/512;
subplot(2,2,4);plot(f,abs(X3)/max(abs(X3)));
title('x3(n)的幅度谱');xlabel('f(Hz)');ylabel('|X3(jf)|');


clear; close all; clc % 清空命令行窗口，清除工作区变量，关闭所有图形窗口

% 定义采样频率
fs = 10000; % 原始信号的采样频率为 10000 Hz
fs1 = 1000; % 第一个抽样频率为 1000 Hz
fs2 = 400; % 第二个抽样频率为 400 Hz
fs3 = 200; % 第三个抽样频率为 200 Hz

% 定义时间序列 t 和信号参数
t = 0:1/fs:0.1; % 时间序列 t 从 0 到 0.1 秒，步长为 1/fs
A = 444.128; % 信号振幅
a = 50 * sqrt(2) * pi; % 衰减系数，50*sqrt(2)*pi
b = a; % 频率系数，等于 a

% 生成模拟信号 xa
xa = exp(-a*t) .* sin(b*t); % 生成模拟信号 xa

% 定义频率和角频率序列
k = 0:511; % 频率索引从 0 到 511
f = fs * k / 512; % 频率序列 f
w = 2 * pi * k / 512; % 角频率序列 w

% 计算模拟信号的傅里叶变换
Xa = xa * exp(-1j * (1:length(xa))' * w); % 计算模拟信号 xa 的傅里叶变换 Xa

% 生成抽样信号 x1
T1 = 1/fs1; % 第一个抽样信号的采样周期
t1 = 0:T1:0.1; % 抽样信号的时间序列
x1 = A * exp(-a .* t1) .* sin(b * t1); % 生成抽样信号 x1
X1 = x1 * exp(-1j * (1:length(x1))' * w); % 计算抽样信号 x1 的傅里叶变换 X1

% 生成抽样信号 x2
T2 = 1/fs2; % 第二个抽样信号的采样周期
t2 = 0:T2:0.1; % 抽样信号的时间序列
x2 = A * exp(-a .* t2) .* sin(b * t2); % 生成抽样信号 x2
X2 = x2 * exp(-1j * (1:length(x2))' * w); % 计算抽样信号 x2 的傅里叶变换 X2

% 生成抽样信号 x3
T3 = 1/fs3; % 第三个抽样信号的采样周期
t3 = 0:T3:0.1; % 抽样信号的时间序列
x3 = A * exp(-a .* t3) .* sin(b * t3); % 生成抽样信号 x3
X3 = x3 * exp(-1j * (1:length(x3))' * w); % 计算抽样信号 x3 的傅里叶变换 X3

% 绘制模拟信号及其频谱
figure(1); % 创建第一个图形窗口
subplot(2,2,1); % 创建 2x2 子图的第一个子图
plot(t, xa); % 绘制模拟信号 xa
axis([0, max(t), min(xa), max(xa)]); % 设置坐标轴范围
title('模拟信号'); % 设置子图标题
xlabel('t(s)'); ylabel('Xa(t)'); % 设置坐标轴标签
line([0, max(t)], [0, 0]); % 绘制 y=0 的水平线

subplot(2,2,2); % 创建 2x2 子图的第二个子图
plot(f, abs(Xa)/max(abs(Xa))); % 绘制模拟信号 xa 的幅度频谱
title('模拟信号的幅度频谱'); % 设置子图标题
axis([0, 500, 0, 1]); % 设置坐标轴范围
xlabel('f(Hz)'); ylabel('|Xa(jf)|'); % 设置坐标轴标签

% 绘制抽样信号 x1 及其频谱
subplot(2,2,3); % 创建 2x2 子图的第三个子图
stem(t1, x1, '.'); % 绘制抽样信号 x1 的样本点
line([0, max(t1)], [0, 0]); % 绘制 y=0 的水平线
axis([0, max(t1), min(x1), max(x1)]); % 设置坐标轴范围
title('抽样序列 x1(n) (fs1=1kHz)'); % 设置子图标题
xlabel('n'); ylabel('X1(n)'); % 设置坐标轴标签

f1 = fs1 * k / 512; % 计算抽样信号 x1 的频率序列
subplot(2,2,4); % 创建 2x2 子图的第四个子图
plot(f1, abs(X1)/max(abs(X1))); % 绘制抽样信号 x1 的幅度频谱
title('x1(n) 的幅度谱'); % 设置子图标题
xlabel('f(Hz)'); ylabel('|X1(jf)|'); % 设置坐标轴标签

% 绘制抽样信号 x2 及其频谱
figure(2); % 创建第二个图形窗口
subplot(2,2,1); % 创建 2x2 子图的第一个子图
stem(t2, x2, '.'); % 绘制抽样信号 x2 的样本点
line([0, max(t2)], [0, 0]); % 绘制 y=0 的水平线
axis([0, max(t2), min(x2), max(x2)]); % 设置坐标轴范围
title('抽样序列 x2(n) (fs2=400Hz)'); % 设置子图标题
xlabel('n'); ylabel('X2(n)'); % 设置坐标轴标签

f2 = fs2 * k / 512; % 计算抽样信号 x2 的频率序列
subplot(2,2,2); % 创建 2x2 子图的第二个子图
plot(f2, abs(X2)/max(abs(X2))); % 绘制抽样信号 x2 的幅度频谱
title('x2(n) 的幅度谱'); % 设置子图标题
xlabel('f(Hz)'); ylabel('|X2(jf)|'); % 设置坐标轴标签

% 绘制抽样信号 x3 及其频谱
subplot(2,2,3); % 创建 2x2 子图的第三个子图
stem(t3, x3, '.'); % 绘制抽样信号 x3 的样本点
line([0, max(t3)], [0, 0]); % 绘制 y=0 的水平线
axis([0, max(t3), min(x3), max(x3)]); % 设置坐标轴范围
title('抽样序列 x3(n) (fs3=200Hz)'); % 设置子图标题
xlabel('n'); ylabel('X3(n)'); % 设置坐标轴标签

f3 = fs3 * k / 512; % 计算抽样信号 x3 的频率序列
subplot(2,2,4); % 创建 2x2 子图的第四个子图
plot(f3, abs(X3)/max(abs(X3))); % 绘制抽样信号 x3 的幅度频谱
title('x3(n) 的幅度谱'); % 设置子图标题
xlabel('f(Hz)'); ylabel('|X3(jf)|'); % 设置坐标轴标签

