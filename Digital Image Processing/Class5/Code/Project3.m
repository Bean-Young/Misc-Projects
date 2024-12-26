img_path = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class5/IMAGES/ckt-board-orig.tif';
output_dir = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class5/Figures/';

img = imread(img_path);
noisy_sp = imnoise(img, 'salt & pepper', 0.25);

figure;
imshow(noisy_sp);
title('Salt-and-Pepper Noise (Pa = Pb = 0.25)');
exportgraphics(gcf, fullfile(output_dir, 'noisy_salt_pepper_0.25.png'), 'Resolution', 300);
close;


% Apply 7 x 7 median filtering
median_filtered = medfilt2(noisy_sp, [7, 7]);

figure;
imshow(median_filtered);
title('7 x 7 Median Filter');
exportgraphics(gcf, fullfile(output_dir, 'median_filtered.png'), 'Resolution', 300);
close;

% Adaptive median filter with maximum subwindow size 7 x 7
adaptive_filtered = adaptive_median_filter(noisy_sp, 7);

figure;
imshow(adaptive_filtered);
title('Adaptive Median Filter (Max Subwindow 7 x 7)');
exportgraphics(gcf, fullfile(output_dir, 'adaptive_median_filtered.png'), 'Resolution', 300);
close;


figure('Name', 'All Results', 'NumberTitle', 'off');
set(gcf, 'Position', [100, 100, 1800, 600]);

subplot(1, 3, 1);
imshow(noisy_sp);
title('Salt-and-Pepper Noise');

subplot(1, 3, 2);
imshow(median_filtered);
title('7 x 7 Median Filter');

subplot(1, 3, 3);
imshow(adaptive_filtered);
title('Adaptive Median Filter');

exportgraphics(gcf, fullfile(output_dir, 'all_results.png'), 'Resolution', 300);