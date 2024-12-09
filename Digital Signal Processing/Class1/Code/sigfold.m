function [y,n] = sigfold(x,n0)
y=fliplr(x);
n=-max(n0):-min(n0);
end