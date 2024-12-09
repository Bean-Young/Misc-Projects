x1=[1,1,1];x2=[2,2,3,0,0,1,4];
y1=circonvtim(x1,x2,7);
y2=circonvtim(x1,x2,10);
subplot(5, 1, 1); stem(y1); axis([0, 10, 0, 20]); title('y_1(n)=x_1(n)⑦x_2(n)');
subplot(5, 1, 2); stem(y2); axis([0, 10, 0, 20]); title('y_1(n)=x_1(n)⑩x_2(n)');