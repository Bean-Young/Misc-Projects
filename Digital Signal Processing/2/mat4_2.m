syms z;
XZ1=z/(z-1);
XZ2=z/(z-1)^3;
X1=iztrans(XZ1)
x2=iztrans(XZ2)
syms z; % 定义符号变量 z
XZ1 = z / (z - 1); % 定义 Z 域信号 XZ1
XZ2 = z / (z - 1)^3; % 定义 Z 域信号 XZ2
X1 = iztrans(XZ1); % 计算 XZ1 的逆 Z 变换，得到时域信号 X1
x2 = iztrans(XZ2); % 计算 XZ2 的逆 Z 变换，得到时域信号 x2
