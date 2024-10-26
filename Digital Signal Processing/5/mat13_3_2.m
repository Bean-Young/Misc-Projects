clc;clear all;close all;
N_values = [10, 20, 50, 100];
wc = 0.25;
M = 512;
f = 0:0.5/M:0.5-0.5/M;
figure;
hold on;
for N = N_values
    h = fir1(N, wc, boxcar(N+1));
    H = freqz(h, 1, M);
    plot(f, abs(H), 'LineWidth', 1.5);
end
hold off;
legend('N=10', 'N=20', 'N=50', 'N=100');
xlabel('\omega/(2\pi)');ylabel('|H(e^{j\omega})|');
axis([0, 0.5, 0, 1.2]);
grid on;
