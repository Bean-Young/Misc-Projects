function Xk = DFTmat(xn)
N = length(xn);
n = 0:N-1; k = n';
WN = exp(-j * 2 * pi / N);
Wnk = WN .^ (k * n);
Xk = xn * Wnk;
