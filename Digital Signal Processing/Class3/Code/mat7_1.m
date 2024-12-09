clear; clc; close all;
n = 0:10; xn = 10 * 0.8.^n;
m = 0:14;
xn1 = [xn, zeros(1, 15 - 11)];
yn1 = xn1(mod(-m, 15) + 1);
subplot(2,2,1); stem(n, xn); ylabel('x(n)'); xlabel('n'); grid on;
subplot(2,2,2); stem(m, yn1); ylabel('x((-n))_{15}'); xlabel('n'); grid on;

yn2 = cirshftt(xn, 6, 15);
yn3 = cirshftt(xn, -4, 15);
subplot(2,2,3); stem(m, yn2); ylabel('x((n-6))_{15}'); xlabel('n'); grid on;
subplot(2,2,4); stem(m, yn3); ylabel('x((n+4))_{15}'); xlabel('n'); grid on;








% 清空工作区变量，清空命令行窗口，关闭所有图形窗口
clear; clc; close all;

% 定义时间序列和信号
n = 0:99; % 时间序列 n 从 0 到 99
x3 = cos(0.48 * pi * n) + cos(0.52 * pi * n); % 定义信号 x3(n)

% 计算信号的长度
N = length(x3); % 信号长度

% 计算信号的离散傅里叶变换 (DFT)
X3 = DFTmat(x3); % 调用自定义 DFTmat 函数计算 DFT
k = n; % 频率索引 k 与时间序列 n 相同
w = 2 * pi * k / N; % 计算频率 w

% 计算幅度响应
magX3 = abs(X3); % 计算幅度响应 |X3(k)|

% 绘制信号 x3(n)
subplot(2,1,1); % 创建 2x1 子图的第一个子图
stem(n, x3); % 使用 stem 函数绘制信号 x3(n)
ylabel('x(n)'); % 设置 y 轴标签
xlabel('n'); % 设置 x 轴标签
grid on; % 显示网格线

% 绘制信号的幅度响应
subplot(2,1,2); % 创建 2x1 子图的第二个子图
plot(w / pi, magX3); % 使用 plot 函数绘制幅度响应
ylabel('|X_3(k)|'); % 设置 y 轴标签
xlabel('\omega (\times \pi)'); % 设置 x 轴标签
grid on; % 显示网格线

% 自定义离散傅里叶变换 (DFT) 函数
function X = DFTmat(x)
    N = length(x); % 信号长度
    n = 0:N-1; % 时间索引
    k = n'; % 频率索引
    WN = exp(-1j * 2 * pi / N); % DFT 矩阵元素
    nk = k * n; % 计算矩阵索引的乘积
    W = WN .^ nk; % 生成 DFT 矩阵
    X = x * W; % 计算 DFT
end
