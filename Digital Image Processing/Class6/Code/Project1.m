% Define input images and their respective slice numbers
imageFiles = {
'/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class6/IMAGES/picker_phantom.tif', 8;
'/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class6/IMAGES/weld-original.tif', 3;
'/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class6/IMAGES/tropical_rain_grayscale.tif', 20
};

% Define the output folder
outputFolder = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class6/Figure';
% Process each image
for i = 1:size(imageFiles, 1)
% Load the current image and slice number
imgFile = imageFiles{i, 1};
numSlice = imageFiles{i, 2};
grayImage = imread(imgFile);

% Generate a colormap for the given slice number
colorMap = parula(numSlice); 
slicedRGB = intensity_slicing(grayImage, numSlice, colorMap);

% Display results side-by-side
figure('Name', ['Results for ', imgFile], 'Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
imshow(grayImage, []);
title(['Original Grayscale Image']);

subplot(1, 2, 2);
imshow(slicedRGB);
title(['Intensity Sliced (', num2str(numSlice), ' slices)']);

% Save the results
[~, imgName, ~] = fileparts(imgFile);
saveFigurePath = fullfile(outputFolder, [imgName, '_comparison.png']);
saveas(gcf, saveFigurePath);

saveImagePath = fullfile(outputFolder, [imgName, '_sliced.png']);
imwrite(slicedRGB, saveImagePath);

% Close the figure
close(gcf);
end