function output = alpha_trimmed_filter(img, window_size, d)
    [rows, cols] = size(img);
    pad_size = floor(window_size / 2); % Padding size for borders
    padded_img = padarray(double(img), pad_size, 'symmetric'); % Symmetric padding
    output = zeros(size(img)); % Initialize output

    % Ensure d is valid
    total_window_size = prod(window_size);
    if d >= total_window_size
        error('Trim value d must be less than the total number of elements in the window');
    end

    % Process each pixel
    for i = 1:rows
        for j = 1:cols
            % Extract the local window
            window = padded_img(i:i+window_size(1)-1, j:j+window_size(2)-1);
            sorted_window = sort(window(:)); % Sort pixel values in the window
            trimmed_window = sorted_window((d/2)+1:end-(d/2)); % Trim d/2 values from both ends
            output(i, j) = mean(trimmed_window); % Calculate the mean of the trimmed window
        end
    end

    output = uint8(output); % Convert back to uint8 format
end