function avg = averageIntensity(image)
    % Ensure the image is in double format for accurate calculation
    image = double(image);
    
    % Calculate the sum of all pixel intensities
    totalIntensity = sum(image(:));
    
    % Calculate the total number of pixels
    numPixels = numel(image);
    
    % Compute the average intensity
    avg = totalIntensity / numPixels;
end