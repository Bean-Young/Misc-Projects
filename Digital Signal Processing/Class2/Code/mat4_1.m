syms w0 n z a
x1=a^n;
x2=n*a^n;
x3=exp(j*w0*n);
X1=ztrans(x1)
X2=ztrans(x2)
X3=ztrans(x3)
syms w0 n z a % 定义符号变量 w0, n, z, a
x1 = a^n; % 定义信号 x1 为 a 的 n 次方
x2 = n * a^n; % 定义信号 x2 为 n 乘以 a 的 n 次方
x3 = exp(j * w0 * n); % 定义信号 x3 为 e^(j * w0 * n)
X1 = ztrans(x1); % 计算 x1 的 Z 变换
X2 = ztrans(x2); % 计算 x2 的 Z 变换
X3 = ztrans(x3); % 计算 x3 的 Z 变换
