% Specify the folder containing the images
imageFolder = '/Users/youngbean/Desktop/Class3/Code/IMAGES';

% List of image filenames
imageFiles = {'darkPollen.jpg', 'lightPollen.jpg', 'lowContrastPollen.jpg', 'pollen.jpg'};

% Output folder for results
outputFolder = '/Users/youngbean/Desktop/Class3/Code/Results';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Loop through each image file
for i = 1:length(imageFiles)
    % Construct the full path of the image
    imagePath = fullfile(imageFolder, imageFiles{i});
    
    % Read the image
    imageData = imread(imagePath);
    
    % Perform histogram equalization
    enhancedImage = histogramEqualization(imageData);
    
    % Save the enhanced image
    enhancedImagePath = fullfile(outputFolder, ['enhanced_', imageFiles{i}]);
    imwrite(enhancedImage, enhancedImagePath);
    
    % Compute histograms
    originalHist = generateHistogram(imageData); % Use previous histogram function
    enhancedHist = generateHistogram(enhancedImage);
    
    % Plot the results
    figure;
    subplot(2, 2, 1);
    imshow(imageData);
    title(['Original Image: ', imageFiles{i}]);
    
    subplot(2, 2, 2);
    bar(originalHist);
    title('Original Histogram');
    xlabel('Pixel Intensity (0-255)');
    ylabel('Frequency');
    
    subplot(2, 2, 3);
    imshow(enhancedImage);
    title('Enhanced Image');
    
    subplot(2, 2, 4);
    bar(enhancedHist);
    title('Enhanced Histogram');
    xlabel('Pixel Intensity (0-255)');
    ylabel('Frequency');
    
    % Save the figure
    saveas(gcf, fullfile(outputFolder, ['comparison_', imageFiles{i}, '.png']));
end