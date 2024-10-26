clear;close all;clc;
A=444.128;a=50*sqrt(2)* pi;b=a;
for k=1:2
if k==1 Fs=400;
elseif k==2 Fs=1000;end
T=1/Fs;dt=T/3;
Tp=0.03;
t=0:dt:Tp;
n=0:Tp/T;
TMN=ones(length(n),1)*t-n'*T*ones(1,length(t));
x=A*exp(-a.*n*T).*sin(b*n*T);
xa=x* sinc(Fs* TMN);
subplot(2,1,k);plot(t, xa);hold on ;
axis([0,max(t),min(xa)-10,max(xa)+10]);
st1=sprintf('由Fs=%d',Fs);st2='Hz抽样序列x(n)重构的信号';
ylabel('x_a(t)');
st=[st1,st2];title(st)
xo=A*exp(-a.*t).*sin(b*t);
stem(t, xo,'.') ;line([0,max(t)],[0 ,0]);
emax2=max(abs(xa-xo))
end

clear; close all; clc; % 清空命令行窗口，清除工作区变量，关闭所有图形窗口

% 定义信号参数
A = 444.128; % 信号振幅
a = 50 * sqrt(2) * pi; % 衰减系数
b = a; % 频率系数，等于 a

% 迭代处理两种不同的采样频率
for k = 1:2
    if k == 1
        Fs = 400; % 第一种采样频率 400 Hz
    elseif k == 2
        Fs = 1000; % 第二种采样频率 1000 Hz
    end
    
    T = 1 / Fs; % 采样周期
    dt = T / 3; % 时间步长，采样周期的三分之一
    Tp = 0.03; % 信号时长 0.03 秒
    t = 0:dt:Tp; % 连续时间序列
    n = 0:Tp / T; % 采样点序列
    TMN = ones(length(n), 1) * t - n' * T * ones(1, length(t)); % 计算时间矩阵
    
    % 生成抽样信号
    x = A * exp(-a .* n * T) .* sin(b * n * T); % 计算离散时间序列的信号 x(n)
    
    % 使用 sinc 函数重构信号
    xa = x * sinc(Fs * TMN); % 利用 sinc 插值函数重构信号 xa(t)
    
    % 绘制重构信号
    subplot(2, 1, k); % 创建 2x1 子图的第 k 个子图
    plot(t, xa); hold on; % 绘制重构信号 xa(t)
    axis([0, max(t), min(xa) - 10, max(xa) + 10]); % 设置坐标轴范围
    
    % 设置标题和标签
    st1 = sprintf('由 Fs = %d ', Fs); % 生成采样频率的字符串
    st2 = 'Hz 抽样序列 x(n) 重构的信号'; % 标题后半部分
    ylabel('x_a(t)'); % 设置 y 轴标签
    st = [st1, st2]; % 组合完整标题
    title(st); % 设置子图标题
    
    % 生成原始信号
    xo = A * exp(-a .* t) .* sin(b * t); % 计算连续时间的原始信号 xo(t)
    
    % 绘制原始信号
    stem(t, xo, '.'); % 使用 stem 函数绘制原始信号 xo(t)
    line([0, max(t)], [0, 0]); % 绘制 y=0 的水平线
    
    % 计算重构误差
    emax2 = max(abs(xa - xo)); % 计算重构信号与原始信号之间的最大误差
end
