%b)
rose = imread('/Users/youngbean/Desktop/Experiments in Digital Image Processing/Class1/Data/Fig_rose.tif');
shrunk_rose = pixel_replication(rose, 16, 'shrink');
figure;
imshow(shrunk_rose);

%c)
zoomed_rose = pixel_replication(shrunk_rose, 16, 'zoom');
figure; 
imshow(zoomed_rose);