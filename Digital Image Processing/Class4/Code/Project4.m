% Load the image
imagePath = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class4/Code/IMAGES/Fig_blurry_moon.tif';
outputDir = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class4/Code/Figure';

originalImage = imread(imagePath);

% Convert to grayscale if needed
if size(originalImage, 3) == 3
    originalImage = rgb2gray(originalImage);
end

% Convert image to double for processing
originalImage = double(originalImage);

% Define the two Laplacian masks
laplacianMask1 = [0 -1 0; -1 5 -1; 0 -1 0];
laplacianMask2 = [-1 -1 -1; -1 9 -1; -1 -1 -1];

% Apply the first Laplacian mask
laplacianFiltered1 = imfilter(originalImage, laplacianMask1, 'same', 'replicate');

% Apply the second Laplacian mask
laplacianFiltered2 = imfilter(originalImage, laplacianMask2, 'same', 'replicate');

% Convert results to uint8 for display and saving
enhancedImage1 = uint8(laplacianFiltered1);
enhancedImage2 = uint8(laplacianFiltered2);

% Save the results
imwrite(enhancedImage1, fullfile(outputDir, 'enhanced_image_mask1.tif'));
imwrite(enhancedImage2, fullfile(outputDir, 'enhanced_image_mask2.tif'));

% Display the results
figure;

subplot(1, 3, 1);
imshow(uint8(originalImage));
title('Original Image');

subplot(1, 3, 2);
imshow(enhancedImage1);
title('Enhanced Image (Mask 1)');

subplot(1, 3, 3);
imshow(enhancedImage2);
title('Enhanced Image (Mask 2)');