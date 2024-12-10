% Load the image
imageData = imread('/Users/youngbean/Desktop/Class3/Code/IMAGES/woman.tif');

% Flip the image vertically
flippedVertically = flipimage(imageData, 'vertical');
imwrite(flippedVertically, '/Users/youngbean/Desktop/Class3/Code/Results/woman_flipped_vertically.png'); % Save the vertically flipped image

% Flip the image horizontally
flippedHorizontally = flipimage(imageData, 'horizontal');
imwrite(flippedHorizontally, '/Users/youngbean/Desktop/Class3/Code/Results/woman_flipped_horizontally.png'); % Save the horizontally flipped image

% Display original, vertically flipped, and horizontally flipped images together
figure; % Open a new figure window

% Display the original image
subplot(1, 3, 1); % Create a subplot in a 1x3 grid, position 1
imshow(imageData); % Display the original image
title('Original Image'); % Add title

% Display the vertically flipped image
subplot(1, 3, 2); % Create a subplot in a 1x3 grid, position 2
imshow(flippedVertically); % Display the vertically flipped image
title('Vertically Flipped Image'); % Add title

% Display the horizontally flipped image
subplot(1, 3, 3); % Create a subplot in a 1x3 grid, position 3
imshow(flippedHorizontally); % Display the horizontally flipped image
title('Horizontally Flipped Image'); % Add title