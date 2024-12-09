function [significant_digits, error] = calculate_yyz(approx_value, true_value)
    % 计算误差
    error = abs(approx_value - true_value);
    
    % 计算整数位数
    integer_digits = count_integer_digits_yyz(approx_value);
    
    % 计算有效数字
    n = 0;
    while error < 0.5 * 10^(-n-1)
        n = n + 1;
    end
    
    % 总有效数字为整数位数 + 小数部分的有效数字
    significant_digits = integer_digits + n;
end

function integer_digits = count_integer_digits_yyz(value)
    % 计算整数部分的位数
    integer_digits = length(num2str(floor(value)));
end