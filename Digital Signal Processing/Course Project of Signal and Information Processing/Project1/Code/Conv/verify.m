% 定义三个信号 x[n], h[n] 和 g[n]
x = [1, 2, 3];
h = [4, 5, 6];
g = [7, 8];

% 定义标准冲激响应函数 δ[n]
n_delta = 0:10;  % 时间范围
delta = [1, zeros(1, length(n_delta)-1)];  % 冲激响应

% 设置图形窗口
figure;

%% 1. 显示原始信号 x[n]
subplot(4, 2, 1);  % 图1：显示信号 x[n]
stem(0:length(x)-1, x, 'b', 'filled');
title('信号 x[n]');
xlabel('n');
ylabel('Amplitude');
grid on;

%% 2. 显示原始信号 h[n]
subplot(4, 2, 2);  % 图2：显示信号 h[n]
stem(0:length(h)-1, h, 'r', 'filled');
title('信号 h[n]');
xlabel('n');
ylabel('Amplitude');
grid on;

%% 3. 显示原始信号 g[n]
subplot(4, 2, 3);  % 图3：显示信号 g[n]
stem(0:length(g)-1, g, 'g', 'filled');
title('信号 g[n]');
xlabel('n');
ylabel('Amplitude');
grid on;

%% 4. 显示标准的冲激响应函数 δ[n]
subplot(4, 2, 4);  % 图4：显示标准冲激响应 δ[n]
stem(n_delta, delta, 'm', 'filled');
title('标准冲激响应 δ[n]');
xlabel('n');
ylabel('Amplitude');
grid on;

%% 5. 交换律验证
subplot(4, 2, 5);  % 图5：验证 x[n] * h[n] 与 h[n] * x[n] 的交换律
conv_xh = conv(x, h);
conv_hx = conv(h, x);
stem(0:length(conv_xh)-1, conv_xh, 'b', 'filled');
hold on;
stem(0:length(conv_hx)-1, conv_hx, 'r--');
title('交换律验证');
xlabel('n');
ylabel('Amplitude');
legend('x[n] * h[n]', 'h[n] * x[n]');
grid on;
hold off;

%% 6. 结合律验证
subplot(4, 2, 6);  % 图6：验证结合律 (x[n] * h[n]) * g[n] = x[n] * (h[n] * g[n])
conv_xh = conv(x, h);
conv_xh_g = conv(conv_xh, g);
conv_hg = conv(h, g);
conv_x_hg = conv(x, conv_hg);
stem(0:length(conv_xh_g)-1, conv_xh_g, 'b', 'filled');
hold on;
stem(0:length(conv_x_hg)-1, conv_x_hg, 'r--');
title('结合律验证');
xlabel('n');
ylabel('Amplitude');
legend('(x[n] * h[n]) * g[n]', 'x[n] * (h[n] * g[n])');
grid on;
hold off;

%% 7. 分配律验证
subplot(4, 2, 7);  % 图7：验证分配律 x[n] * (h[n] + g[n]) = x[n] * h[n] + x[n] * g[n]
% 将 g[n] 进行零填充，使其与 h[n] 长度一致
g_padded = [g, zeros(1, length(h) - length(g))];  % g[n] 零填充
conv_xh = conv(x, h);  % x[n] * h[n]
conv_xg = conv(x, g_padded);  % x[n] * g[n]
conv_x_h_plus_g = conv(x, h + g_padded);  % x[n] * (h[n] + g[n])
conv_xh_plus_xg = conv_xh + conv_xg;  % x[n] * h[n] + x[n] * g[n]
stem(0:length(conv_x_h_plus_g)-1, conv_x_h_plus_g, 'b', 'filled');
hold on;
stem(0:length(conv_xh_plus_xg)-1, conv_xh_plus_xg, 'r--');
title('分配律验证');
xlabel('n');
ylabel('Amplitude');
legend('x[n] * (h[n] + g[n])', 'x[n] * h[n] + x[n] * g[n]');
grid on;
hold off;

%% 8. 冲激响应验证
subplot(4, 2, 8);  % 图8：h[n] 与 δ[n] 的卷积验证
% 将 h[n] 进行零填充，以匹配 δ[n] 的长度
h_padded = [h, zeros(1, length(delta) - length(h))];  % h[n] 零填充
conv_h_delta = conv(h_padded, delta);  % h[n] 与 δ[n] 的卷积
stem(0:length(conv_h_delta)-1, conv_h_delta, 'g', 'filled');
hold on;
stem(0:length(h)-1, h, 'r--');
title('h[n] 与 δ[n] 的卷积验证');
xlabel('n');
ylabel('Amplitude');
legend('h[n] * δ[n]', 'h[n]');
grid on;
hold off;

% 调整图像布局
sgtitle('卷积定理验证');