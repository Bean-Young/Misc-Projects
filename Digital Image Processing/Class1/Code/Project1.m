%a)
% Create a 512x512 black image
image = zeros(512, 512, 'uint8');
rect_height = 20; % Height of the rectangle (20 pixels)
rect_width = 40;  % Width of the rectangle (40 pixels)
center_row = 512 / 2;
center_col = 512 / 2;
row_start = round(center_row - rect_height / 2);
row_end = round(center_row + rect_height / 2 - 1);
col_start = round(center_col - rect_width / 2);
col_end = round(center_col + rect_width / 2 - 1);
image(row_start:row_end, col_start:col_end) = 255;
figure;
imshow(image);
%b)
imwrite(image, '/Users/youngbean/Desktop/Experiments in Digital Image Processing/Class1/Programming/test.bmp');
%c)
I = imread('test.bmp');
figure;
imshow(I);
%e)
imwrite(I, 'test.tif');
imwrite(I, 'test.jpg');
info_bmp = dir('test.bmp');
info_tif = dir('test.tif');
info_jpg = dir('test.jpg');
% Display file sizes
fprintf('BMP Size: %d bytes\n', info_bmp.bytes);
fprintf('TIF Size: %d bytes\n', info_tif.bytes);
fprintf('JPG Size: %d bytes\n', info_jpg.bytes);
%f)
copyfile('test.bmp', '/Users/youngbean/Desktop/Experiments in Digital Image Processing/Class1/Data/test.bmp');
movefile('test.tif', '/Users/youngbean/Desktop/Experiments in Digital Image Processing/Class1/Data/test.tif');
movefile('test.jpg', '/Users/youngbean/Desktop/Experiments in Digital Image Processing/Class1/Data/test.jpg');