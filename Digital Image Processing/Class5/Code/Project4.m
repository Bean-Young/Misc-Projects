img_path = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class5/IMAGES/top_ left_flower.tif';
output_dir = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class5/Figures/';

rgb_img = imread(img_path);

figure;
imshow(rgb_img);
title('Original RGB Image');
exportgraphics(gcf, fullfile(output_dir, 'original_rgb_image.png'), 'Resolution', 300);
close;

% (b) Convert to gray image using I = (R + G + B) / 3
% Separate the Red, Green, and Blue channels
R = double(rgb_img(:, :, 1));
G = double(rgb_img(:, :, 2));
B = double(rgb_img(:, :, 3));

% Calculate intensity
I_simple = (R + G + B) / 3;
I_simple = uint8(I_simple); 
figure;
imshow(I_simple);
title('Gray Image (Simple Conversion)');
exportgraphics(gcf, fullfile(output_dir, 'gray_image_simple.png'), 'Resolution', 300);
close;

% (c) Adjust proportions of R, G, B and compare results
% Define different proportions
proportions = [
    0.3, 0.59, 0.11;  % Common grayscale conversion weights
    0.4, 0.4, 0.2;    % Another example
    0.33, 0.33, 0.34; % Balanced
    0.5, 0.3, 0.2;    % Stronger weight on R
    0.2, 0.5, 0.3     % Stronger weight on G
];

% Initialize storage for results
gray_images = cell(size(proportions, 1) + 1, 1);

% Add simple conversion result as the first image
gray_images{1} = I_simple;

% Process and save results
for i = 1:size(proportions, 1)
    % Get current proportions
    r_weight = proportions(i, 1);
    g_weight = proportions(i, 2);
    b_weight = proportions(i, 3);
    
    % Calculate intensity with weighted proportions
    I_weighted = r_weight * R + g_weight * G + b_weight * B;
    I_weighted = uint8(I_weighted); % Convert back to uint8 for display
    
    % Store the result
    gray_images{i + 1} = I_weighted;
    
    % Save each image separately
    figure;
    imshow(I_weighted);
    title(sprintf('R: %.2f, G: %.2f, B: %.2f', r_weight, g_weight, b_weight));
    exportgraphics(gcf, fullfile(output_dir, sprintf('gray_image_r%.2f_g%.2f_b%.2f.png', r_weight, g_weight, b_weight)), 'Resolution', 300);
    close;
end

figure('Name', 'Comparison of Gray Images with Sample Conversion', 'NumberTitle', 'off');
set(gcf, 'Position', [100, 100, 1400, 800]);

subplot(2, ceil((size(proportions, 1) + 1) / 2), 1);
imshow(gray_images{1});
title('Simple Conversion (R = G = B = 1/3)');

for i = 1:size(proportions, 1)
    subplot(2, ceil((size(proportions, 1) + 1) / 2), i + 1);
    imshow(gray_images{i + 1});
    title(sprintf('R: %.2f, G: %.2f, B: %.2f', proportions(i, 1), proportions(i, 2), proportions(i, 3)));
end

exportgraphics(gcf, fullfile(output_dir, 'gray_image_comparison.png'), 'Resolution', 300);