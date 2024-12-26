img = imread('/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class5/IMAGES/ckt-board-orig.tif'); 
output_dir = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class5/Figures/';

d_values = [2, 6, 10, 14, 20];
results = {
    'Original Image', img;
    'Salt & Pepper Noise', imnoise(img, 'salt & pepper', 0.1);
    'Median Filter (Salt & Pepper)', medfilt2(imnoise(img, 'salt & pepper', 0.1), [5, 5]);
    'Salt & Pepper + Gaussian Noise', imnoise(imnoise(img, 'salt & pepper', 0.1), 'gaussian', 0, 0.01);
};

h = fspecial('average', [5, 5]);
results{end+1, 1} = 'Arithmetic Mean Filter';
results{end, 2} = imfilter(results{4, 2}, h, 'replicate');

% Add Geometric Mean Filter
geo_filtered = exp(imfilter(log(double(results{4, 2}) + 1), ones(5, 5), 'replicate')) .^ (1 / 25);
results{end+1, 1} = 'Geometric Mean Filter';
results{end, 2} = uint8(geo_filtered - 1);

% Add Median Filter (Gaussian)
results{end+1, 1} = 'Median Filter (Gaussian)';
results{end, 2} = medfilt2(results{4, 2}, [5, 5]);

% Add Alpha-Trimmed Filters for Different d Values
for i = 1:numel(d_values)
    d_current = d_values(i);
    results{end+1, 1} = sprintf('Alpha-Trimmed Mean Filter (d = %d)', d_current);
    results{end, 2} = alpha_trimmed_filter(results{4, 2}, [5, 5], d_current);
end


for i = 1:size(results, 1)
    image_name = results{i, 1};
    current_image = results{i, 2};
    
    % Save image with title
    figure;
    imshow(current_image, []);
    title(image_name, 'Interpreter', 'none');
    exportgraphics(gcf, fullfile(output_dir, sprintf('%s_image.png', strrep(image_name, ' ', '_'))), 'Resolution', 300);
    close;

    % Save histogram with title
    figure;
    imhist(current_image);
    title(sprintf('Histogram of %s', image_name), 'Interpreter', 'none');
    exportgraphics(gcf, fullfile(output_dir, sprintf('%s_histogram.png', strrep(image_name, ' ', '_'))), 'Resolution', 300);
    close;
end

num_images = size(results, 1);
rows = ceil(num_images / 4); % Adjust rows for a max of 4 columns

% Figure for all images
figure('Name', 'All Processed Images', 'NumberTitle', 'off');
set(gcf, 'Position', [100, 100, 1800, 1200]); % Adjust figure size

for i = 1:num_images
    subplot(rows, 4, i);
    imshow(results{i, 2}, []);
    title(results{i, 1}, 'Interpreter', 'none');
end

exportgraphics(gcf, fullfile(output_dir, 'all_images_displayed.png'), 'Resolution', 300);

figure('Name', 'All Histograms', 'NumberTitle', 'off');
set(gcf, 'Position', [100, 100, 1800, 1200]); % Adjust figure size

for i = 1:num_images
    subplot(rows, 4, i);
    imhist(results{i, 2});
    title(sprintf('Histogram of %s', results{i, 1}), 'Interpreter', 'none');
end

exportgraphics(gcf, fullfile(output_dir, 'all_histogram_displayed.png'), 'Resolution', 300);

% Extract original image
original_image = results{1, 2}; % Original image

% Collect images for comparison
comparison_images = { ...
    'Median Filter (Gaussian)', results{7, 2}; % Median Filter
    sprintf('Alpha-Trimmed (d = %d)', d_values(1)), results{8, 2};
    sprintf('Alpha-Trimmed (d = %d)', d_values(2)), results{9, 2};
    sprintf('Alpha-Trimmed (d = %d)', d_values(3)), results{10, 2};
    sprintf('Alpha-Trimmed (d = %d)', d_values(4)), results{11, 2};
    sprintf('Alpha-Trimmed (d = %d)', d_values(5)), results{12, 2};
};

% Compute differences with the original image
diff_images = cell(size(comparison_images, 1), 1);
for i = 1:size(comparison_images, 1)
    diff_images{i} = imabsdiff(original_image, comparison_images{i, 2});
end

figure('Name', 'Differences: Filters vs Original', 'NumberTitle', 'off');
set(gcf, 'Position', [100, 100, 1800, 800]); % Adjust figure size

rows = 2; 
cols = 3; 

for i = 1:size(diff_images, 1)
    subplot(rows, cols, i);
    imshow(diff_images{i}, []);
    title(sprintf('Difference: %s', comparison_images{i, 1}), 'Interpreter', 'none');
end

% Save the combined difference image
exportgraphics(gcf, fullfile(output_dir, 'all_difference_images.png'), 'Resolution', 300);
close;

% Save individual difference images
for i = 1:size(diff_images, 1)
    figure;
    imshow(diff_images{i}, []);
    title(sprintf('Difference: %s', comparison_images{i, 1}), 'Interpreter', 'none');
    exportgraphics(gcf, fullfile(output_dir, sprintf('difference_%s.png', strrep(comparison_images{i, 1}, ' ', '_'))), 'Resolution', 300);
    close;
end

original_image = results{1, 2};
original_histogram = imhist(original_image);

% Collect histograms for comparison
comparison_images = { ...
    'Median Filter (Gaussian)', results{7, 2}; % Median Filter
    sprintf('Alpha-Trimmed (d = %d)', d_values(1)), results{8, 2};
    sprintf('Alpha-Trimmed (d = %d)', d_values(2)), results{9, 2};
    sprintf('Alpha-Trimmed (d = %d)', d_values(3)), results{10, 2};
    sprintf('Alpha-Trimmed (d = %d)', d_values(4)), results{11, 2};
    sprintf('Alpha-Trimmed (d = %d)', d_values(5)), results{12, 2};
};

% Compute histogram differences
histogram_differences = cell(size(comparison_images, 1), 1);
for i = 1:size(comparison_images, 1)
    comparison_histogram = imhist(comparison_images{i, 2});
    histogram_differences{i} = (original_histogram - comparison_histogram);
end

figure('Name', 'Histogram Differences: Filters vs Original', 'NumberTitle', 'off');
set(gcf, 'Position', [100, 100, 1800, 800]); % Adjust figure size

rows = 2; 
cols = 3;

for i = 1:size(histogram_differences, 1)
    subplot(rows, cols, i);
    bar(histogram_differences{i}, 'k'); % Use a bar plot to show histogram difference
    title(sprintf('Histogram Difference: %s', comparison_images{i, 1}), 'Interpreter', 'none');
    xlabel('Pixel Intensity');
    ylabel('Difference');
end

exportgraphics(gcf, fullfile(output_dir, 'all_histogram_differences.png'), 'Resolution', 300);
close;

for i = 1:size(histogram_differences, 1)
    figure;
    bar(histogram_differences{i}, 'k');
    title(sprintf('Histogram Difference: %s', comparison_images{i, 1}), 'Interpreter', 'none');
    xlabel('Pixel Intensity');
    ylabel('Difference');
    exportgraphics(gcf, fullfile(output_dir, sprintf('histogram_difference_%s.png', strrep(comparison_images{i, 1}, ' ', '_'))), 'Resolution', 300);
    close;
end