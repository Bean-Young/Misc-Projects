% 实验人：杨跃浙
% 圆周率真实值
pi_true = 3.141592653589793;

% 近似值 1
pi_approx_1 = 3.1415;
[significant_digits_1, error_1] = calculate_yyz(pi_approx_1, pi_true);
fprintf('π* = %.7f 的有效数字位数: %d, 误差限: %.7e\n', pi_approx_1, significant_digits_1, error_1);

% 近似值 2
pi_approx_2 = 3.141593;
[significant_digits_2, error_2] = calculate_yyz(pi_approx_2, pi_true);
fprintf('π* = %.7f 的有效数字位数: %d, 误差限: %.7e\n', pi_approx_2, significant_digits_2, error_2);

% 可选：如果需要将结果保存为变量，可如下存储
results = struct( ...
    'pi_approx_1', struct('value', pi_approx_1, 'significant_digits', significant_digits_1, 'error_limit', error_1), ...
    'pi_approx_2', struct('value', pi_approx_2, 'significant_digits', significant_digits_2, 'error_limit', error_2) ...
);
