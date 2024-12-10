function histogramArray = generateHistogram(imageData)
    % generateHistogram Generates the histogram of an image.
    % Parameters:
    %   imageData - An image array with pixel values in the range 0-255.
    % Returns:
    %   histogramArray - A 1x256 array containing the histogram values.
    
    % Convert image to grayscale if it is not already
    if size(imageData, 3) == 3
        imageData = rgb2gray(imageData); % Convert to grayscale
    end

    % Initialize histogram array
    histogramArray = zeros(1, 256);

    % Calculate histogram
    for intensity = 0:255
        histogramArray(intensity + 1) = sum(imageData(:) == intensity);
    end
end