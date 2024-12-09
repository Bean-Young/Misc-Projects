function fz=caiyang(fy,fs)
fs0=10000;tp=0.1;
t=[-tp:1/fs0:tp];
k1=0:999;k2=-999:-1;
m1=length(k1);m2=length(k2),
f=[fs0* k2/m2,fs0 * k1/m1];
w=[-2* pi* k2/m2,2*pi* k1/m1];
fx1=eval(fy);
FX1=fx1 * exp(-j* [1:length(fx1)]'* w);
figure;
subplot(2,1,1);plot(t,fx1 ,'r');
title('原信号');xlabel('时间 t/s');
axis([min(t),max(t),min(fx1),max(fx1)]);
subplot(2,1,2);plot(f,abs(FX1),'r')
title('原信号幅度频谱');xlabel('频率 f/Hz');
axis([-100,100,0,max(abs(FX1))+5]);
Ts=1/fs;
t1=-tp:Ts:tp;
f1=[fs * k2/m2,fs * k1/m1];
t=t1;
fz=eval(fy);
FZ=fz * exp(-j* [1:length(fz)]' * w);
figure;
subplot(2,1,1);stem(t,fz,'.');
title('抽样信号');xlabel('时间 t/s');
line([min(t) ,max(t)],[0,0]);
subplot(2,1,2);plot(f1 ,abs(FZ),'m');
title('抽样信号幅度频谱');xlabel('频率 f/Hz');