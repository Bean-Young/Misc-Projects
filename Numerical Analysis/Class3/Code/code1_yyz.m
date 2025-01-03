% 任务 1
% 实验人: 杨跃浙
xdata = [1, 2, 4, 5]; % x 数据
ydata = [3, 6, 2, 1]; % y 数据
x = 2.4; % 要插值的点

% 调用 Newton 插值函数
[result1, result2, diff_table] = newton_yyz(xdata, ydata, x);

% 获取差商表的大小
[num_rows, num_cols] = size(diff_table);

% 打印差商表（加表头）
fprintf('差商表:\n');
fprintf('%10s', 'x_i'); % 打印第一列标题
fprintf('%10s', 'f(x_i)'); % 打印第二列标题
for order = 1:(num_cols - 1)
    fprintf('%10s', sprintf('Δ^%d', order)); % 动态生成差商列标题
end
fprintf('\n');

% 打印差商表数据
for i = 1:num_rows
    fprintf('%10.4f', xdata(i)); % 打印当前的 x_i 值
    fprintf('%10.4f', diff_table(i, 1)); % 打印 f(x_i) 值，从 diff_table 获取
    for order = 2:num_cols
        if (order - 1) <= (num_rows - i)
            val = diff_table(i, order);
            % 检查差商值是否为空或 NaN，若是则填充 0
            if isempty(val) || isnan(val)
                val = 0;
            end
            fprintf('%10.4f', val);
        else
            fprintf('%10.4f', 0); % 填充 0 以保持对齐
        end
    end
    fprintf('\n');
end

% 输出插值结果
fprintf('一次插值多项式在 x = %.2f 处的近似值为: %.4f\n', x, result1);
fprintf('二次插值多项式在 x = %.2f 处的近似值为: %.4f\n', x, result2);

