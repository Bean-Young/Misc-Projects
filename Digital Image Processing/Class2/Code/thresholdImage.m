function thresholdedImage = thresholdImage(image, threshold)
    % Ensure the image is in double format for accurate comparison
    image = double(image);
    
    % Generate the thresholded image
    % Pixels greater than the threshold are set to 1, others are set to 0
    thresholdedImage = image > threshold;
end