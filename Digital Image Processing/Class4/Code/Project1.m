% Define the path to the original image
imagePath = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class4/Code/IMAGES/Fig_test_pattern_blurring_orig.tif';

% Define mask sizes
maskSizes = [3, 5, 9, 15, 35];

% Define output directory
outputDir = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class4/Code/Figure';

% Read the image
originalImage = imread(imagePath);

% Display the original image
figure('Name', 'Original and Averaged Images', 'NumberTitle', 'off');
subplot(2, 3, 1);
imshow(originalImage, []);
title('Original Image');

filteredImages = cell(length(maskSizes), 1);

for idx = 1:length(maskSizes)
    maskSize = maskSizes(idx);  
    % Create an averaging filter using fspecial
    avgFilter = fspecial('average', [maskSize maskSize]);
    
    % Apply the filter using imfilter with 'replicate' to handle borders
    filteredImage = imfilter(originalImage, avgFilter, 'replicate');
    
    % Store the filtered image
    filteredImages{idx} = filteredImage;
    
    % Define the output filename
    outputFilename = fullfile(outputDir, sprintf('filtered_%dx%d.tif', maskSize, maskSize));
    
    % Save the filtered image
    imwrite(filteredImage, outputFilename);
    fprintf('Saved filtered image with %dx%d mask as %s\n', maskSize, maskSize, outputFilename);
    
    % Display the filtered image
    subplot(2, 3, idx + 1);
    imshow(filteredImage, []);
    title(sprintf('Average Filter %dx%d', maskSize, maskSize));
end

% Add a super title
sgtitle('Spatial Averaging (Mean) Filtering with Different Mask Sizes');