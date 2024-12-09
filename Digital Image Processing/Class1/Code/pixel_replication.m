%a)
function resized_image = pixel_replication(image, factor, mode)
    if strcmp(mode, 'shrink')
        resized_image = image(1:factor:end, 1:factor:end);
    elseif strcmp(mode, 'zoom')
        resized_image = repelem(image, factor, factor);
    end
end