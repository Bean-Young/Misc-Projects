% 定义时间轴
n1 = 0:10;
n2 = 0:5;

% 定义信号 x[n]
x = sin(0.2*pi*n1);

% 定义信号 h[n]
h = cos(0.4*pi*n2);
% 计算线性卷积 y[n]
y = conv(x, h);
% 定义卷积结果的时间轴
ny = (n1(1)+n2(1)):(n1(end)+n2(end));

% 绘制信号 x[n]
subplot(3,1,1);
stem(n1, x, 'filled');
title('信号 x[n]');
xlabel('n');
ylabel('x[n]');

% 绘制信号 h[n]
subplot(3,1,2);
stem(n2, h, 'filled');
title('信号 h[n]');
xlabel('n');
ylabel('h[n]');

% 绘制卷积结果 y[n]
subplot(3,1,3);
stem(ny, y, 'filled');
title('线性卷积 y[n] = x[n] * h[n]');
xlabel('n');
ylabel('y[n]');