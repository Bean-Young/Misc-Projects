% Read the input image
Moon = imread('/Users/youngbean/Desktop/Experiments in Digital Image Processing/Class2/Data/Fig_blurry_moon.tif');

% Calculate the average intensity using the averageIntensity function
ave = averageIntensity(Moon);

% Apply the thresholdImage function using the average intensity
ThresholdedMoon = thresholdImage(Moon, ave);

% Display the original and thresholded images
figure;
subplot(1, 2, 1);
imshow(Moon);
title('Original Image');

subplot(1, 2, 2);
imshow(ThresholdedMoon);
title(['Thresholded Image (Threshold = ', num2str(ave), ')']);