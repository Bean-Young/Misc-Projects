clear; clc; close all;
n = 0:99; x3 = cos(0.48 * pi * n) + cos(0.52 * pi * n);
N = length(x3);
X3 = DFTmat(x3);
k = n; w = 2 * pi * k / N;
magX3 = abs(X3);
subplot(2,1,1); stem(n, x3); ylabel('x(n)'); xlabel('n'); grid on;
subplot(2,1,2); plot(w / pi, magX3); ylabel('|X_3(k)|'); xlabel('omega(x pi)'); grid on;
