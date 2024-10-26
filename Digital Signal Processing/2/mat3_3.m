f1='sin(2*pi*60*t)+cos(2*pi*25*t)+cos(2* pi*30 *t)';
fs0=caiyang(f1,80);
fr0=huifu(fs0,80);
fs1=caiyang(f1,120)
fr1=huifu(fs1,120);
fs2=caiyang(f1,150);
fr2=huifu(fs2,150);

% 清空工作区变量，关闭所有图形窗口，清空命令行窗口
clear; close all; clc;

% 定义信号 f1
f1 = 'sin(2*pi*60*t) + cos(2*pi*25*t) + cos(2*pi*30*t)';

% 使用采样频率 80 Hz 进行采样和重构
fs0 = caiyang(f1, 80); % 采样
fr0 = huifu(fs0, 80); % 重构

% 使用采样频率 120 Hz 进行采样和重构
fs1 = caiyang(f1, 120); % 采样
fr1 = huifu(fs1, 120); % 重构

% 使用采样频率 150 Hz 进行采样和重构
fs2 = caiyang(f1, 150); % 采样
fr2 = huifu(fs2, 150); % 重构
