n=-5:5;x=(-0.9).^n;
k=-200:200;w=(pi/100)*k;
X=x*(exp(-j*pi/100)).^(n'*k);
magX=abs(X);angX=angle(X);
subplot(2,1,1);plot(w/pi,magX);grid on;
axis([-2,2,0,15]);xlabel('\omega(x\pi)');ylabel('幅度|H(e^j^\omega)|');
subplot(2,1,2);plot(w/pi,angX/pi);grid on;
axis([-2,2,-1,1]); xlabel('\omega(x\pi)');ylabel('相位（弧度/\pi)');
