% 加载音频文件
[audio, fs] = audioread('mix.wav');
disp(fs);

% 设置时间向量
t = (0:length(audio)-1) / fs;

% 添加高斯白噪声
noisy_audio = audio + 0.03 * randn(size(audio)); % 调整噪声水平

% 步骤1：定义滤波器指标
passband_cutoff = 3000;        % 通带截止频率（Hz）
stopband_cutoff = 5000;        % 阻带截止频率（Hz）
passband_ripple = 0.5;         % 通带最大衰减（dB）
stopband_attenuation = 70;     % 阻带最小衰减（dB）

% 过渡带宽
transition_band = stopband_cutoff - passband_cutoff;

% 使用凯撒窗的阻带衰减要求
A = stopband_attenuation;       % 阻带衰减需求
delta_f = transition_band / fs; % 归一化过渡带宽
order = ceil((A - 8) / (2.285 * 2 * pi * delta_f)); % 凯撒窗的经验公式计算阶数
if mod(order, 2) == 0
    order = order + 1; % 确保阶数为奇数
end
disp(['滤波器阶数: ', num2str(order)]);

% 设置凯撒窗的 beta 参数（根据衰减需求）
beta = 0.1102 * (A - 8.7);

% 理想低通滤波器的冲激响应
n = 0:order;
center = floor(order / 2);
h_low = sin(2 * pi * passband_cutoff * (n - center) / fs) ./ (pi * (n - center));
h_low(center + 1) = 2 * passband_cutoff / fs; % 修正中心点，避免除以0

% 使用凯撒窗
window_kaiser = kaiser(order + 1, beta)'; % 创建凯撒窗
h_low_windowed = h_low .* window_kaiser; % 应用凯撒窗

% 使用 freqz 计算频率响应以获取阻带和通带信息
[H, f_response] = freqz(h_low_windowed, 1, 1024, fs);

% 查找通带和阻带截止频率
passband_end = passband_cutoff; % 通带截止频率假设为截止频率
stopband_start = stopband_cutoff; % 阻带起始频率
passband_idx = find(f_response <= passband_end);
stopband_idx = find(f_response >= stopband_start);

% 计算实际通带截止和阻带起始
passband_cutoff_actual = f_response(passband_idx(end)); % 实际通带截止频率
stopband_cutoff_actual = f_response(stopband_idx(1));   % 实际阻带起始频率

% 计算通带和阻带衰减
passband_gain_max = max(abs(H(passband_idx))); % 通带最大增益
stopband_gain_max = max(abs(H(stopband_idx))); % 阻带最大增益
passband_attenuation_actual = -20 * log10(passband_gain_max); % 通带实际衰减
stopband_attenuation_actual = -20 * log10(stopband_gain_max); % 阻带实际衰减

% 输出通带和阻带信息
disp(['通带截止频率: ', num2str(passband_cutoff_actual), ' Hz']);
disp(['阻带截止频率: ', num2str(stopband_cutoff_actual), ' Hz']);
disp(['通带衰减: ', num2str(passband_attenuation_actual), ' dB']);
disp(['阻带衰减: ', num2str(stopband_attenuation_actual), ' dB']);

% 对加噪信号进行滤波
filtered_audio = filter(h_low_windowed, 1, noisy_audio);

% 计算频谱
Y = fft(audio);
Y_noisy = fft(noisy_audio);
Y_filtered = fft(filtered_audio);
f = (0:length(Y)-1) * (fs / length(Y)); % 频率向量

% 绘制时域和频域图在一张图上
figure;

% 时域图
subplot(2,3,1);
plot(t, audio);
title('原始音频 - 时域');
xlabel('时间 (s)');
ylabel('幅度');

subplot(2,3,2);
plot(t, noisy_audio);
title('加噪音频 - 时域');
xlabel('时间 (s)');
ylabel('幅度');

subplot(2,3,3);
plot(t, filtered_audio);
title('滤波后音频 - 时域');
xlabel('时间 (s)');
ylabel('幅度');

% 频域图
subplot(2,3,4);
plot(f, abs(Y));
title('原始音频 - 频域');
xlabel('频率 (Hz)');
ylabel('幅度');

subplot(2,3,5);
plot(f, abs(Y_noisy));
title('加噪音频 - 频域');
xlabel('频率 (Hz)');
ylabel('幅度');

subplot(2,3,6);
plot(f, abs(Y_filtered));
title('滤波后音频 - 频域');
xlabel('频率 (Hz)');
ylabel('幅度');

% 绘制低通滤波器的频率响应图
% 将频率归一化，范围从 0 到 1 表示从 0 到 π
normalized_freq = f_response / (fs / 2); % 归一化频率至 \omega / \pi

% 绘制低通滤波器的频率响应图
figure;
plot(normalized_freq, 20*log10(abs(H)));
title('低通FIR滤波器的频率响应（凯泽窗）');
xlabel('\omega / \pi');
ylabel('幅度 (dB)');

% 顺序播放原始、加噪和滤波后音频
disp('播放原始音频...');
sound(audio, fs);
pause(length(audio)/fs + 1); % 播放完成后暂停1秒

disp('播放加噪音频...');
sound(noisy_audio, fs);
pause(length(noisy_audio)/fs + 1); % 播放完成后暂停1秒

disp('播放滤波后音频...');
sound(filtered_audio, fs);