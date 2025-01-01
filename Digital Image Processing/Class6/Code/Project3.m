I = imread('/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class6/IMAGES/Fig_strawberries.tif');  
I = im2double(I);               

% 定义输出文件夹
outputFolder = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class6/Figure';

% 2. 定义参数
% 目标中心颜色 a
a = [0.6863, 0.1608, 0.1922];  % (R,G,B)
% 立方体半边长 W/2
W  = 0.2549;   % 这里指的是整个立方体的边长
% 球体半径
R0 = 0.1765;  
% 替换色
outColor = [0.5, 0.5, 0.5];

% 拆出 R/G/B 通道
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);

% 3. 立方体切片 (公式 6.5-7)
% 若对任何通道 i 有 |rᵢ - aᵢ| > W/2，则该像素“在立方体外”
maskCube = (abs(R - a(1)) > W/2) | ...
            (abs(G - a(2)) > W/2) | ...
            (abs(B - a(3)) > W/2);

% 将立方体外的像素替换成 outColor(0.5,0.5,0.5)
I_cube = I;  % 复制原图
for c = 1:3
    channelC = I_cube(:,:,c);
    channelC(maskCube) = outColor(c);
    I_cube(:,:,c) = channelC;
end

% 4. 球体切片 (公式 6.5-8)
% 若 (R - a_R)^2 + (G - a_G)^2 + (B - a_B)^2 > R0^2，则像素“在球体外”
distSq = (R - a(1)).^2 + (G - a(2)).^2 + (B - a(3)).^2;
maskSphere = distSq > R0^2;

% 将球体外的像素替换成 outColor
I_sphere = I;
for c = 1:3
    channelC = I_sphere(:,:,c);
    channelC(maskSphere) = outColor(c);
    I_sphere(:,:,c) = channelC;
end

% 5. 显示并保存结果
figure('Name','Color Slicing Demo','Position',[100,100,1500,400]);
subplot(1,3,1); imshow(I);        title('Original');
subplot(1,3,2); imshow(I_cube);   title('Cube-based Slicing');
subplot(1,3,3); imshow(I_sphere); title('Sphere-based Slicing');

outCube   = fullfile(outputFolder, 'strawberries_cube.png');
outSphere = fullfile(outputFolder, 'strawberries_sphere.png');

% 保存图像
imwrite(I_cube,   outCube);
imwrite(I_sphere, outSphere);

disp('Color slicing demo finished. Results saved to:');
disp(outCube);
disp(outSphere);
