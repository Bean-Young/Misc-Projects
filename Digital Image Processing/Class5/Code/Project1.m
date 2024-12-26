img = imread('/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class5/IMAGES/ckt-board-orig.tif'); 

mean = 0;
variance = 0.01;
noisy_gaussian = imnoise(img, 'gaussian', mean, variance);

salt_pepper_density = 0.05; % Density of salt-and-pepper noise (proportion of noisy pixels)
noisy_salt_pepper = imnoise(img, 'salt & pepper', salt_pepper_density);

output_dir = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class5/Figures/';


figure;
imshow(img);
title('Original Image');
exportgraphics(gcf, fullfile(output_dir, 'original_image.png'), 'Resolution', 300);
close;


figure;
imshow(noisy_gaussian);
title('Gaussian Noise');
exportgraphics(gcf, fullfile(output_dir, 'noisy_gaussian.png'), 'Resolution', 300);
close;

figure;
imshow(noisy_salt_pepper);
title('Salt-and-Pepper Noise');
exportgraphics(gcf, fullfile(output_dir, 'noisy_salt_pepper.png'), 'Resolution', 300);
close;


figure;
subplot(1, 3, 1);
imshow(img);
title('Original Image');

subplot(1, 3, 2);
imshow(noisy_gaussian);
title('Gaussian Noise');

subplot(1, 3, 3);
imshow(noisy_salt_pepper);
title('Salt-and-Pepper Noise');

exportgraphics(gcf, fullfile(output_dir, 'all_images_displayed.png'), 'Resolution', 300);