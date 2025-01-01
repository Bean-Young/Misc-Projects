function hsiImage = rgb2hsi(rgbImage)
    % Convert an RGB image to HSI color space.
    rgbImage = im2double(rgbImage);
    R = rgbImage(:, :, 1);
    G = rgbImage(:, :, 2);
    B = rgbImage(:, :, 3);

    % Calculate Intensity
    I = (R + G + B) / 3;

    % Calculate Saturation
    minRGB = min(cat(3, R, G, B), [], 3);
    S = 1 - (minRGB ./ I);
    S(I == 0) = 0; % Handle division by zero

    % Calculate Hue
    num = 0.5 * ((R - G) + (R - B));
    den = sqrt((R - G).^2 + (R - B) .* (G - B));
    H = acosd(num ./ (den + eps));
    H(B > G) = 360 - H(B > G);
    H = H / 360; % Normalize to [0, 1]

    % Combine into HSI image
    hsiImage = cat(3, H, S, I);
end