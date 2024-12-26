img_path = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class5/IMAGES/Fig_strawberries.tif';
output_dir = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class5/Figures/';

rgb_img = imread(img_path);

figure;
imshow(rgb_img);
title('Original RGB Image');
exportgraphics(gcf, fullfile(output_dir, 'original_rgb_image.png'), 'Resolution', 300);
close;


% (a) Suppress intensity in RGB model
k = 0.7; % Suppression factor

% Suppress intensity in RGB model
rgb_suppressed = uint8(k * double(rgb_img)); % Scale intensity


figure;
imshow(rgb_suppressed);
title('Intensity Suppressed in RGB Model');
exportgraphics(gcf, fullfile(output_dir, 'intensity_suppressed_rgb.png'), 'Resolution', 300);
close;

% (b) Suppress intensity in CMY model
% Convert RGB to CMY
cmy_img = 1 - im2double(rgb_img); % Normalize to [0, 1] range and invert

% Suppress intensity in CMY model
cmy_suppressed = cmy_img * k;

% Convert back to RGB
rgb_from_cmy = uint8((1 - cmy_suppressed) * 255);

figure;
imshow(rgb_from_cmy);
title('Intensity Suppressed in CMY Model');
exportgraphics(gcf, fullfile(output_dir, 'intensity_suppressed_cmy.png'), 'Resolution', 300);
close;

figure('Name', 'Comparison of Intensity Suppression', 'NumberTitle', 'off');
set(gcf, 'Position', [100, 100, 1600, 600]);

subplot(1, 3, 1);
imshow(rgb_img);
title('Original RGB Image');

subplot(1, 3, 2);
imshow(rgb_suppressed);
title('Suppressed in RGB Model');

subplot(1, 3, 3);
imshow(rgb_from_cmy);
title('Suppressed in CMY Model');

exportgraphics(gcf, fullfile(output_dir, 'intensity_suppression_comparison.png'), 'Resolution', 300);
