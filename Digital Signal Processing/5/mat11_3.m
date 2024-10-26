clear;close all;clc;
N=3;T=250*10^(-6);
fs=1/T;fc=1000;
[B,A]=butter(N,2*pi*fc,'s');
[num1,den1]=impinvar(B,A,fs)
[h1,w]=freqz(num1,den1);
[B,A]=butter(N,2/T*tan(2*pi*fc*T/2),'s');
[num2,den2]=bilinear(B,A,fs)
[h2,w]=freqz(num2,den2);
f=w/pi*2000;
plot(f,abs(h1),'k',f,abs(h2),'b--');
grid on;
xlabel('频率(Hz)');ylabel('幅值(dB)');
legend('冲击响应不变法','双线性变换法');
