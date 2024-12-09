clc; clear all;
Fs = 20 * 10^3; fp = 4000; fst = 4500;
wp = 2 * pi * fp / Fs; ws = 2 * pi * fst / Fs; Rp = 0.5; As = 60;
wc = (wp + ws) / 2;
delw = ws - wp;
N = ceil(11 * pi / delw); M = N - 1;
n = [0:N-1];
h = fir1(M, wc / pi, blackman(N));
[db, mag, pha, grd, w] = freqz_m(h, [1]);
dw = 2 * pi / 1000;
subplot(311)
stem(n, h, '.', 'linewidth', 2); title('布莱克曼窗'); xlabel('n'); ylabel('w(n)'); axis([0, N, 0, 0.45]); grid
subplot(312)
plot(w / pi, db, 'linewidth', 2);
title('幅度响应(dB)'); xlabel('\omega/\pi'); ylabel('20log|H(e^{j\omega})| (dB)'); axis([0, 1, -120, 10]); grid
set(gca, 'xtickmode', 'manual', 'xtick', [0, 0.2, 0.4, 0.45, 0.7, 1.0]);
set(gca, 'ytickmode', 'manual', 'ytick', [-120, -90, -60, 0, 10]);
subplot(313)
plot(w / pi, pha, 'linewidth', 2); axis([0, 1, -4, 4]); grid
title('相位响应'); xlabel('\omega/\pi'); ylabel('arg20log|H(e^{j\omega})|');
