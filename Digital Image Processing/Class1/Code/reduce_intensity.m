%a)
function reduced_image = reduce_intensity(image, levels)
    image = uint8(image);
    factor = 256 / levels;
    reduced_image = uint8(floor(double(image) / factor) * factor);
    reduced_image(reduced_image == (256 - factor)) = 255;
end