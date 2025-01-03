function y_piecewise = piecewise_interpolation_yyz(x, y, x_fine)
    y_piecewise = interp1(x, y, x_fine, 'linear');
end