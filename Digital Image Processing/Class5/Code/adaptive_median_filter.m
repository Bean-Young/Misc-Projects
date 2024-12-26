function output = adaptive_median_filter(img, max_window_size)
    [rows, cols] = size(img);
    output = img; % Initialize output image
    half_max_size = floor(max_window_size / 2);

    for i = 1:rows
        for j = 1:cols
            window_size = 3; % Start with a 3x3 window
            while window_size <= max_window_size
                half_size = floor(window_size / 2);

                % Get subwindow
                r_min = max(i - half_size, 1);
                r_max = min(i + half_size, rows);
                c_min = max(j - half_size, 1);
                c_max = min(j + half_size, cols);
                subwindow = img(r_min:r_max, c_min:c_max);

                % Compute Zmin, Zmax, Zmed
                Zmin = min(subwindow(:));
                Zmax = max(subwindow(:));
                Zmed = median(subwindow(:));

                % Check Level A
                if Zmed > Zmin && Zmed < Zmax
                    % Check Level B
                    if img(i, j) > Zmin && img(i, j) < Zmax
                        output(i, j) = img(i, j); % Keep original pixel
                    else
                        output(i, j) = Zmed; % Replace with median
                    end
                    break; % Window size is sufficient
                else
                    window_size = window_size + 2; % Expand the window
                end
            end
        end
    end
end