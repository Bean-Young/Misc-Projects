function rgbImage = hsi2rgb(hsiImage)
    % Convert an HSI image to RGB color space.
    H = hsiImage(:, :, 1) * 360; % Convert to degrees
    S = hsiImage(:, :, 2);
    I = hsiImage(:, :, 3);

    % Initialize
    [rows, cols] = size(H);
    R = zeros(rows, cols);
    G = zeros(rows, cols);
    B = zeros(rows, cols);

    % H in [0, 120)
    idx = (H >= 0 & H < 120);
    B(idx) = I(idx) .* (1 - S(idx));
    R(idx) = I(idx) .* (1 + S(idx) .* cosd(H(idx)) ./ cosd(60 - H(idx)));
    G(idx) = 3 * I(idx) - (R(idx) + B(idx));

    % H in [120, 240)
    idx = (H >= 120 & H < 240);
    H(idx) = H(idx) - 120;
    R(idx) = I(idx) .* (1 - S(idx));
    G(idx) = I(idx) .* (1 + S(idx) .* cosd(H(idx)) ./ cosd(60 - H(idx)));
    B(idx) = 3 * I(idx) - (R(idx) + G(idx));

    % H in [240, 360)
    idx = (H >= 240 & H <= 360);
    H(idx) = H(idx) - 240;
    G(idx) = I(idx) .* (1 - S(idx));
    B(idx) = I(idx) .* (1 + S(idx) .* cosd(H(idx)) ./ cosd(60 - H(idx)));
    R(idx) = 3 * I(idx) - (G(idx) + B(idx));

    % Combine
    rgbImage = cat(3, R, G, B);
    rgbImage = im2uint8(rgbImage); % Scale back to [0, 255]
end