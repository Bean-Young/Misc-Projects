clear; clc; close all;
x1 = [1, 1, 1]; x2 = [2,2,3,0,0,1,4];
y_lin = conv(x1, x2);
y1 = circonvtim(x1, x2, 7);
y2 = circonvtim(x1, x2, 8);
y3 = circonvtim(x1, x2, 9);
y4 = circonvtim(x1, x2, 10);

subplot(5, 1, 1); stem(y_lin); axis([0, 10, 0, 20]); title('y(n) = x_1(n) * x_2(n)');
subplot(5, 1, 2); stem(y1); axis([0, 10, 0, 20]); title('y_1(n)=x_1(n)⑦x_2(n)');
subplot(5, 1, 3); stem(y2); axis([0, 10, 0, 20]); title('y_1(n)=x_1(n)⑧x_2(n)');
subplot(5, 1, 4); stem(y3); axis([0, 10, 0, 20]); title('y_1(n)=x_1(n)⑨x_2(n)');
subplot(5, 1, 5); stem(y4); axis([0, 10, 0, 20]); title('y_1(n)=x_1(n)⑩x_2(n)');

