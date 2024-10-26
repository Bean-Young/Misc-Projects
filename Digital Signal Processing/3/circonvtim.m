function y = circonvtim(x1, x2, N)
y = zeros(1, N);
x1 = [x1, zeros(1, N - length(x1))];
x2 = [x2, zeros(1, N - length(x2))];
n = 0:N-1;
x3 = x2(mod(-n, N) + 1);
for m = 0:N-1
    x4 = cirshftt(x3, m, N);
    x5 = x1 .* x4;
    y(m + 1) = sum(x5);
end
end
