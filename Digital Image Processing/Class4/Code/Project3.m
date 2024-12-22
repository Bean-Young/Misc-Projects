% Load the image with salt-and-pepper noise
imagePath = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class4/Code/IMAGES/ckt_board_saltpep.tif';
outputDir = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class4/Code/Figure';

% Load the original image
originalImage = imread(imagePath);

% Convert to grayscale if the image is RGB
if size(originalImage, 3) == 3
    originalImage = rgb2gray(originalImage);
end

% Convert to double for filter processing
originalImage = double(originalImage);

% 3x3 Averaging Filter
averagingFilter = fspecial('average', [3 3]);  % Create a 3x3 averaging filter
averagedImage = imfilter(originalImage, averagingFilter, 'same');  % Apply the filter

% Save the result of the averaging filter
imwrite(uint8(averagedImage), fullfile(outputDir, 'averaging_filtered_image.tif'));

% 3x3 Median Filter
medianFilteredImage = medfilt2(originalImage, [3 3]);  % Apply the median filter

% Save the result of the median filter
imwrite(uint8(medianFilteredImage), fullfile(outputDir, 'median_filtered_image.tif'));

% Display the original and filtered images
figure;
subplot(1, 3, 1);
imshow(uint8(originalImage));  % Show the original noisy image
title('Original Image with Salt and Pepper Noise');

subplot(1, 3, 2);
imshow(uint8(averagedImage));  % Show the image after applying averaging filter
title('Averaging Filter Result');

subplot(1, 3, 3);
imshow(uint8(medianFilteredImage));  % Show the image after applying median filter
title('Median Filter Result');
