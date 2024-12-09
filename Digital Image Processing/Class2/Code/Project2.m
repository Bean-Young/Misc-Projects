% Read the original image
image = imread('/Users/youngbean/Desktop/Experiments in Digital Image Processing/Class2/Data/Fig_fractured_spine.tif');

% Normalize the image to the range [0, 1]
image = double(image) / 255; % Convert to double and normalize

% Define the range for parameters c and gamma
c_values = [0.5, 1, 2, 3]; % Scaling factors
gamma_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2,1.4,1.6,1.8,2.0]; % Gamma values

% Create a new figure for displaying results
figure;

% Display the original image in a larger subplot
subplot(2, 1, 1); % Use the top half of the figure for the original image
imshow(image);
title('Original Image', 'FontSize', 14); % Larger title for better emphasis

% Calculate the number of subplots needed for enhanced images
num_c = length(c_values);
num_gamma = length(gamma_values);
total_images = num_c * num_gamma;

% Create a tiled layout for enhanced images (adjusting rows/cols)
cols = num_gamma; % Columns match the number of gamma values
rows = num_c;     % Rows match the number of c values
figure;
subplot(2, 1, 2); % Use the bottom half for the tiled layout
t = tiledlayout(rows, cols, 'TileSpacing', 'compact', 'Padding', 'compact');

% Add row and column labels for better comparison
for r = 1:rows
    for g = 1:cols
        % Compute the current c and gamma values
        c = c_values(r);
        gamma = gamma_values(g);
        
        % Apply the power-law transformation
        transformed_image = c * (image .^ gamma);
        
        % Plot the transformed image in the corresponding tile
        nexttile;
        imshow(transformed_image);
        title(['c=', num2str(c), ', \gamma=', num2str(gamma)], 'FontSize', 10);
    end
end

% Add global labels for rows and columns
xlabel(t, 'Gamma Values (\gamma)', 'FontSize', 12);
ylabel(t, 'Scaling Factors (c)', 'FontSize', 12);

% Add a global title
sgtitle('Power-law Transformations: Original and Enhanced Comparisons', 'FontSize', 16);