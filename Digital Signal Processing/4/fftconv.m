function y=fftconv(xl ,x2,N)
Xk1=fft(xl,N);
Xk2=fft(x2,N);
YK=Xk1.*Xk2;
y=ifft(YK);
