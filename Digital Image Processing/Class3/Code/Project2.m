% Specify the folder containing the images
imageFolder = '/Users/youngbean/Desktop/Class3/Code/IMAGES';

% Specify the folder to save the results
outputFolder = '/Users/youngbean/Desktop/Class3/Code/Results';

% Create the results folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% List of image filenames
imageFiles = {'darkPollen.jpg', 'lightPollen.jpg', 'lowContrastPollen.jpg', 'pollen.jpg'};

% Loop through each image file and generate histogram
for i = 1:length(imageFiles)
    % Construct the full path of the image
    imagePath = fullfile(imageFolder, imageFiles{i});
    
    % Read the image
    imageData = imread(imagePath);
    
    % Generate the histogram
    histogramArray = generateHistogram(imageData);
    
    % Display the histogram
    figure; % Open a new figure window for each image
    bar(histogramArray); % Display the histogram as a bar graph
    title(['Histogram of ', imageFiles{i}]); % Add title to the histogram
    xlabel('Pixel Intensity (0-255)'); % Label for x-axis
    ylabel('Frequency'); % Label for y-axis
    
    % Save the histogram as an image
    outputFileName = fullfile(outputFolder, ['Histogram_', erase(imageFiles{i}, '.jpg'), '.png']);
    saveas(gcf, outputFileName); % Save the figure as a PNG file
end