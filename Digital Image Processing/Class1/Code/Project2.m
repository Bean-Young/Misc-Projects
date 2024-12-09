% Read the first image
I = imread('/Users/youngbean/Desktop/Experiments in Digital Image Processing/Class1/Data/Fig_blurry_moon.tif');
% Create the negative image for the first image
negativeImage = imcomplement(I);
% Read the second image
testImage = imread('/Users/youngbean/Desktop/Experiments in Digital Image Processing/Class1/Data/test.tif');
% Create the negative image for the second image
negativeTest = imcomplement(testImage);

figure;
subplot(1, 2, 1); imshow(I); title('Original - Fig_blurry_moon');
subplot(1, 2, 2); imshow(negativeImage); title('Negative - Fig_blurry_moon');
figure;
subplot(1, 2, 1); imshow(testImage); title('Original - test');
subplot(1, 2, 2); imshow(negativeTest); title('Negative - test');