function slicedRGB = intensity_slicing(grayImage, numSlice, colorMap)
    grayDouble = double(grayImage);
    
    % Define intensity thresholds for slicing
    grayMin = min(grayDouble(:));
    grayMax = max(grayDouble(:));
    thresholds = linspace(grayMin, grayMax, numSlice + 1);
    
    % Initialize RGB output image
    [rows, cols] = size(grayImage);
    slicedRGB = zeros(rows, cols, 3);
    
    % Map each intensity range to corresponding color
    for i = 1:numSlice
        mask = (grayDouble >= thresholds(i)) & (grayDouble < thresholds(i + 1));
        for c = 1:3
            slicedRGB(:, :, c) = slicedRGB(:, :, c) + mask * colorMap(i, c);
        end
    end
end