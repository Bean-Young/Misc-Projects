clc;clear;close all
K=1024;
conv_time=zeros(1,K);
fft_time=zeros(1,K);
for L=1:K
    tc=0;tf=0;
    N=2*L-1;
    nu=ceil(log10(N)/log10(2));N1=2^nu;
    for I=1:100
        x1=rand(1,L);x2=randn(1,L);
        t0=clock;y1=conv(x1,x2);
        t1=etime(clock,t0);tc=tc+t1;
        t0=clock;Y2=fft(x1,N1).*fft(x2,N1);y2=ifft(Y2,N1);
        t2=etime(clock,t0);tf=tf+t2;
    end
    conv_time(L)=tc/100;
    fft_time(L)=tf/100;
end
n=1:K;
plot(n,conv_time(n),'k--');
hold on
plot(n,fft_time(n),'b--');
hold off; xlabel('N');ylabel('t/s');