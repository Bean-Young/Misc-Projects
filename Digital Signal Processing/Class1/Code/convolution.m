function [y,ny] = convolution(x1,nx1,x2,nx2)
N1=length(x1);N2=length(x2);N=N1+N2-1;
nyb=nx1(1)+nx2(1);
nye=nx1(N1)+nx2(N2);
ny=nyb:nye;
x2=fliplr(x2);
M=N1+2*N2-2;
x1=[zeros(1,N2-1),x1,zeros(1,N2-1)];
x2=[x2,zeros(1,N-1)];
for n=0:M-1
    x4=[zeros(1,n),x2(1,1:M-n)];
    x5=x1.*x4;
    y(n+1)=sum(x5);
end
y=y(1,1:N)