% Read the original image
image1 = imread("/Users/youngbean/Desktop/Experiments in Digital Image Processing/Class2/Data/Fig_DFT_no_log.tif");

% Normalize the original image to the range [0, 1]
% Convert the image to double type and divide by 255 to map pixel values to [0, 1]
image1 = double(image1) / 255;

% Define the range for c (different scaling factors for log transformation)
c = [0.1, 0.5, 1, 1.5, 2, 3, 5, 10]; % List of c values to test
num_c = length(c); % Get the total number of c values

% Create a new figure window for displaying results
figure;

% Display the original image in the first subplot
subplot(ceil((num_c + 1) / 3), 3, 1); % The first subplot is reserved for the original image
imshow(image1); % Display the normalized original image
title('Original Image'); % Set the title for the original image

% Loop through all c values and display the log-transformed results
for i = 1:num_c
    % Apply the log transform with the current c value
    % 9 * image1 amplifies the normalized image before applying the log
    % log10 is used for base-10 logarithm, and +1 ensures no log(0) occurs
    image2 = c(i) * log10(9 * image1 + 1);
    
    % Display the transformed image in the subplot
    subplot(ceil((num_c + 1) / 3), 3, i + 1); % Place the result in the appropriate subplot
    imshow(image2); % Display the transformed image
    xlabel(['c = ' num2str(c(i))]); % Label the subplot with the current c value
end

% Add a global title for the figure to describe the experiment
sgtitle('Original Image and Log Transform with Different c Values');