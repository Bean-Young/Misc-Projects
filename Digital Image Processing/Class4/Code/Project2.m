imagePath = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class4/Code/IMAGES/embedded_square_noisy.tif';
outputDir = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class4/Code/Figure';

% Load image
originalImage = imread(imagePath);

% Convert to grayscale if the image is RGB
if ndims(originalImage) == 3
    originalImage = rgb2gray(originalImage);
end

% Local Histogram Equalization (3x3 Neighborhood)
rows = length(originalImage(:,1));
cols = length(originalImage(1,:));
localEqualizedImage = originalImage;

for i = 2:rows-1
    for j = 2:cols-1
        % Extract the 3x3 neighborhood
        neighborhood = originalImage(i-1:i+1, j-1:j+1);
        
        % Calculate the histogram of the neighborhood
        hist_vals = imhist(neighborhood);
        
        % Calculate the cumulative distribution function (CDF)
        cdf = cumsum(hist_vals) / sum(hist_vals);
        
        % Apply histogram equalization to the neighborhood
        for m = 1:3
            for n = 1:3
                original_val = neighborhood(m, n);
                equalized_val = round(cdf(original_val + 1) * 255);
                neighborhood(m, n) = equalized_val;
            end
        end
        
        % Assign the equalized value to the center pixel of the neighborhood
        localEqualizedImage(i, j) = neighborhood(2, 2);
    end
end

% Global Histogram Equalization
globalEqualizedImage = histeq(originalImage);

% Convert the images to uint8 and ensure the range is 0-255 for saving
localEqualizedImage = uint8(localEqualizedImage);
globalEqualizedImage = uint8(globalEqualizedImage);

% Save the output images
imwrite(localEqualizedImage, fullfile(outputDir, 'local_equalized_image.tif'));
imwrite(globalEqualizedImage, fullfile(outputDir, 'global_equalized_image.tif'));

% Display the results
figure;

subplot(1, 3, 1);
imshow(originalImage);
title('Original Image');

subplot(1, 3, 2);
imshow(localEqualizedImage);
title('Local Histogram Equalization');

subplot(1, 3, 3);
imshow(globalEqualizedImage);  
title('Global Histogram Equalization');
