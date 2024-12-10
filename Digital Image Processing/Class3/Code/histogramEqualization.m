function enhancedImage = histogramEqualization(imageData)
    % histogramEqualization Performs histogram equalization on a grayscale image.
    % Parameters:
    %   imageData - Grayscale image data (pixel values in 0-255 range).
    % Returns:
    %   enhancedImage - Histogram-equalized grayscale image.
    
    % Convert to grayscale if image is RGB
    if size(imageData, 3) == 3
        imageData = rgb2gray(imageData);
    end
    
    % Calculate the histogram
    histArray = zeros(1, 256);
    for intensity = 0:255
        histArray(intensity + 1) = sum(imageData(:) == intensity);
    end
    
    % Compute the cumulative distribution function (CDF)
    cdf = cumsum(histArray) / numel(imageData);
    
    % Map the original pixel values to new values based on the CDF
    equalized = uint8(cdf(imageData + 1) * 255);
    
    % Return the equalized image
    enhancedImage = equalized;
end