inputImageFile = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class6/IMAGES/Fig_strawberries.tif'; 
outputFolder = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class6/Figure'; 

% Read the input image
rgbImage = imread(inputImageFile);

% RGB Complement
rgbComplement = 255 - rgbImage;

% Convert RGB to HSI
hsiImage = rgb2hsi(rgbImage);

% HSI Complement
% Complement Hue by shifting 180 degrees (0.5 in normalized range)
hsiComplement = hsiImage;
hsiComplement(:, :, 1) = mod(hsiComplement(:, :, 1) + 0.5, 1); % Hue
hsiComplement(:, :, 2) = hsiComplement(:, :, 2);           % Saturation
hsiComplement(:, :, 3) = 1 - hsiComplement(:, :, 3);           % Intensity

% Convert back to RGB
hsiComplementRGB = hsi2rgb(hsiComplement);

% Display Results
figure('Name', 'Color Complement Results', 'Position', [100, 100, 1200, 800]);

% Original Image
subplot(1, 3, 1);
imshow(rgbImage);
title('Original Image');

% RGB Complement
subplot(1, 3, 2);
imshow(rgbComplement);
title('RGB Complement');

% HSI Complement
subplot(1, 3, 3);
imshow(hsiComplementRGB);
title('HSI Complement');

% Save Results
imwrite(rgbComplement, fullfile(outputFolder, 'strawberries_rgb_complement.jpg'));
imwrite(hsiComplementRGB, fullfile(outputFolder, 'strawberries_hsi_complement.jpg'));