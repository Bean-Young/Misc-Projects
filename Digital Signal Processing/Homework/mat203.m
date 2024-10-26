clc; clear all;
f1=90; f2=130; f3=320; fs=600;
N=64;
n=[0:1:N-1];
k=[0:1:N-1];
xn=sin(2*pi*n*f1/fs) + 1.3*sin(2*pi*n*f2/fs) + 1.6*sin(2*pi*n*f3/fs);
WN=exp(-j*2*pi/N);
nk=n'*k; WNnk=WN.^nk;
Xk=abs(xn*WNnk);
stem(k,Xk,'.','linewidth',2);grid
xlabel('k');title('x(n)的幅度谱');ylabel('|X(k)|');axis([0,63,0,50]);
