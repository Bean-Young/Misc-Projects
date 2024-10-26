clc;clear;close all;
xn=[0,0,ones(1,99)];
hn=[0,0,2,0,0.6,0.3,-0.7];
y=conv(xn,hn);
stem(0:length(y)-1,y);
grid on;xlabel('n');ylabel('系统的响应');
