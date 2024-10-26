clear;clc;close all
N=21;wc=pi/4;
n=0:N-1;a=(N-1)/2;
na=n-a+eps*((n-a)==0);
hdn=sin(wc*na)/pi./na;
if rem(N,2)~=0 hdn(a+1)=wc/pi; end
wn1=boxcar(N);
hn1=hdn.*wn1';
figure(1);
stem(n,hn1,'.');stem(n,hn1,'.');line([0,20],[0,0]);grid on;
xlabel('n');ylabel('h(n)');title('矩形窗设计的 h(n)');
hw1=fft(hn1,512);w1=2*[0:511]/512;
figure(2);
subplot(2,1,1);plot(w1,20*log10(abs(hw1)));grid on;
ylabel('幅度(dB)');title('幅度特性');
subplot(2,1,2);plot(w1,unwrap(angle(hw1)));grid on;
xlabel('\omega/\pi');ylabel('相位(度)');title('相位特性');