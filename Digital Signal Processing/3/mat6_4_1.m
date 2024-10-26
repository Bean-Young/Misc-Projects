clear; clc; close all;
n = 0:99; x = cos(0.48 * pi * n) + cos(0.52 * pi * n);
n1 = 0:9; x1 = x(1:10); N = length(x1);
X1 = DFTfor(x1);
k = n1; w = 2 * pi * k / N;
magX1 = abs(X1);
subplot(2,1,1); stem(n1, x1); ylabel('x(n)'); xlabel('n'); grid on;
subplot(2,1,2); stem(w / pi, magX1); ylabel('|X_1_0(k)|'); xlabel('omega(x\pi)'); grid on;





% 清空工作区变量，清空命令行窗口，关闭所有图形窗口
clear; clc; close all;

% 定义时间序列和信号
n = 0:99; % 时间序列 n 从 0 到 99
x = cos(0.48 * pi * n) + cos(0.52 * pi * n); % 定义信号 x(n)

% 截取前 10 个样本
n1 = 0:9; % 时间序列 n1 从 0 到 9
x1 = x(1:10); % 截取信号 x 的前 10 个样本
N = length(x1); % 计算截取信号的长度

% 计算截取信号的离散傅里叶变换 (DFT)
X1 = DFTfor(x1); % 调用自定义 DFTfor 函数计算 DFT
k = n1; % 频率索引 k 与时间序列 n1 相同
w = 2 * pi * k / N; % 计算频率 w

% 计算幅度响应
magX1 = abs(X1); % 计算幅度响应 |X1(k)|

% 绘制截取信号 x1(n)
subplot(2,1,1); % 创建 2x1 子图的第一个子图
stem(n1, x1); % 使用 stem 函数绘制信号 x1(n)
ylabel('x(n)'); % 设置 y 轴标签
xlabel('n'); % 设置 x 轴标签
grid on; % 显示网格线

% 绘制截取信号的幅度响应
subplot(2,1,2); % 创建 2x1 子图的第二个子图
stem(w / pi, magX1); % 使用 stem 函数绘制幅度响应
ylabel('|X_{10}(k)|'); % 设置 y 轴标签
xlabel('\omega (\times \pi)'); % 设置 x 轴标签
grid on; % 显示网格线

% 自定义离散傅里叶变换 (DFT) 函数
function X = DFTfor(x)
    N = length(x); % 信号长度
    X = zeros(1, N); % 初始化 DFT 结果
    for k = 0:N-1 % 频率索引从 0 到 N-1
        for n = 0:N-1 % 时间索引从 0 到 N-1
            X(k+1) = X(k+1) + x(n+1) * exp(-1j * 2 * pi * k * n / N); % 计算 DFT
        end
    end
end
