%b)
%ct_skull = uint8(repmat(0:255, [256, 1]));
ct_skull = imread('/Users/youngbean/Desktop/Experiments in Digital Image Processing/Class1/Data/Fig_ctskull-256.tif');
levels = 256;
num_reductions = log2(256);
figure;
for i = 0:num_reductions
    % Calculate current levels (256, 128, ..., 2)
    current_levels = 256 / (2^i);
    reduced_image = reduce_intensity(ct_skull, current_levels);
    subplot(3, 3, i + 1); % Arrange in a 3x3 grid
    imshow(reduced_image);
    title(['Levels: ', num2str(current_levels)]);
end
% Display the original image in the last subplot
subplot(3, 3, num_reductions + 1);
imshow(ct_skull);
title('Original Image');
% Reduce the intensity levels of the image to the user-specified value
answer = inputdlg('Enter the desired number of intensity levels (must be a power of 2 between 2 and 256):', ...
                  'Input Intensity Levels', [1 50]);
user_levels = str2double(answer{1});
user_reduced_image = reduce_intensity(ct_skull, user_levels);
output_filename = ['reduced_image_levels_', num2str(user_levels), '.png'];
imwrite(user_reduced_image, output_filename);
fprintf('Image reduced to %d levels saved as: %s\n', user_levels, output_filename);
figure;
subplot(1, 2, 1);
imshow(ct_skull, [0 255]); 
title('Original Image');
subplot(1, 2, 2);
imshow(user_reduced_image, [0 255]);
title(['Reduced to ', num2str(user_levels), ' Levels']);