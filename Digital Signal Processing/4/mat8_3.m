clc; clear; close all;
Nmax = 256;
ditfft_time = zeros(1, Nmax);
for n = 1:Nmax
    x = rand(1, n);
    t = clock;
    ditfft(x);
    ditfft_time(n) = etime(clock, t);
end
k = 1:Nmax;
subplot(3, 1, 1); plot(k, ditfft_time, '--');
ylabel('t/s'); title('DIT-FFT 执行时间');

DFTfor_time = zeros(1, Nmax);
for n = 1:Nmax
    x = rand(1, n);
    t = clock;
    DFTfor(x);
    DFTfor_time(n) = etime(clock, t);
end
subplot(3, 1, 2); plot(k, DFTfor_time, '--');
ylabel('t/s'); title('DFTfor 执行时间');

DFTmat_time = zeros(1, Nmax);
for n = 1:Nmax
    x = rand(1, n);
    t = clock;
    DFTmat(x);
    DFTmat_time(n) = etime(clock, t);
end
subplot(3, 1, 3); plot(k, DFTmat_time, '--');
xlabel('N'); ylabel('t/s'); title('DFTmat 执行时间');
