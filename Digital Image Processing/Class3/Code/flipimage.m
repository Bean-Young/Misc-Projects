function flippedImage = flipimage(imageData, flipDirection)
    % flipImage Flips an image vertically or horizontally.
    % Parameters:
    %   imageData - A matrix storing the image data
    %   flipDirection - A string, either 'vertical' or 'horizontal'
    % Returns:
    %   flippedImage - The flipped image matrix

    if strcmpi(flipDirection, 'vertical')
        flippedImage = flipud(imageData); % Flip the image vertically
    elseif strcmpi(flipDirection, 'horizontal')
        flippedImage = fliplr(imageData); % Flip the image horizontally
    else
        error('Invalid flip direction. Use "vertical" or "horizontal".');
    end
end