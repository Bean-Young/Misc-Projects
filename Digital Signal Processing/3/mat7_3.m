clear; clc; close all;
x1 = [1, 1, 1]; x2 = [1, 2, 3,4,5];
y_lin = conv(x1, x2);
y1 = circonvtim(x1, x2, 5);
y2 = circonvtim(x1, x2, 6);
y3 = circonvtim(x1, x2, 7);
y4 = circonvtim(x1, x2, 8);

subplot(5, 1, 1); stem(y_lin); axis([1, 8, 0, 20]); title('y(n) = x_1(n) * x_2(n)');
subplot(5, 1, 2); stem(y1); axis([1, 8, 0, 20]); title('y_1(n)=x_1(n)⑤x_2(n)');
subplot(5, 1, 3); stem(y2); axis([1, 8, 0, 20]); title('y_1(n)=x_1(n)⑥x_2(n)');
subplot(5, 1, 4); stem(y3); axis([1, 8, 0, 20]); title('y_1(n)=x_1(n)⑦x_2(n)');
subplot(5, 1, 5); stem(y4); axis([1, 8, 0, 20]); title('y_1(n)=x_1(n)⑧x_2(n)');







% 清空工作区变量，清空命令行窗口，关闭所有图形窗口
clear; clc; close all;

% 定义输入信号
x1 = [1, 1, 1]; % 信号 x1
x2 = [1, 2, 3, 4, 5]; % 信号 x2

% 计算线性卷积
y_lin = conv(x1, x2);

% 计算不同长度的循环卷积
y1 = circonvtim(x1, x2, 5); % 长度为 5 的循环卷积
y2 = circonvtim(x1, x2, 6); % 长度为 6 的循环卷积
y3 = circonvtim(x1, x2, 7); % 长度为 7 的循环卷积
y4 = circonvtim(x1, x2, 8); % 长度为 8 的循环卷积

% 绘制线性卷积结果
subplot(5, 1, 1); stem(y_lin); axis([1, 8, 0, 20]);
title('y(n) = x_1(n) * x_2(n)');
xlabel('n'); ylabel('y(n)');

% 绘制长度为 5 的循环卷积结果
subplot(5, 1, 2); stem(y1); axis([1, 8, 0, 20]);
title('y_1(n)=x_1(n) \circledcirc_5 x_2(n)');
xlabel('n'); ylabel('y_1(n)');

% 绘制长度为 6 的循环卷积结果
subplot(5, 1, 3); stem(y2); axis([1, 8, 0, 20]);
title('y_1(n)=x_1(n) \circledcirc_6 x_2(n)');
xlabel('n'); ylabel('y_2(n)');

% 绘制长度为 7 的循环卷积结果
subplot(5, 1, 4); stem(y3); axis([1, 8, 0, 20]);
title('y_1(n)=x_1(n) \circledcirc_7 x_2(n)');
xlabel('n'); ylabel('y_3(n)');

% 绘制长度为 8 的循环卷积结果
subplot(5, 1, 5); stem(y4); axis([1, 8, 0, 20]);
title('y_1(n)=x_1(n) \circledcirc_8 x_2(n)');
xlabel('n'); ylabel('y_4(n)');

% 自定义循环卷积函数
function y = circonvtim(x1, x2, N)
    % CIRCOVTIM 计算信号 x1 和 x2 的循环卷积，结果长度为 N
    % 输入:
    %   x1 - 第一个输入信号
    %   x2 - 第二个输入信号
    %   N  - 循环卷积结果的长度
    % 输出:
    %   y  - 循环卷积结果

    % 将输入信号扩展到长度为 N
    x1 = [x1, zeros(1, N - length(x1))];
    x2 = [x2, zeros(1, N - length(x2))];
    
    % 计算循环卷积
    y = zeros(1, N); % 初始化结果
    for n = 1:N
        for k = 1:N
            y(n) = y(n) + x1(k) * x2(mod(n - k, N) + 1);
        end
    end
end
