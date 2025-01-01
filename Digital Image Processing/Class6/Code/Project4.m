% 设置输入、输出路径
imgFile = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class6/IMAGES/Fig0637(a)(caster_stand_original).tif';
outputFolder = '/Users/youngbean/Documents/Github/Misc-Projects/Digital Image Processing/Class6/Figure';


% 读入图像并转 double [0,1]
I_rgb = imread(imgFile);
I_rgb = im2double(I_rgb);

% 转到 HSV，获取 H/S/V
I_hsv = rgb2hsv(I_rgb);
H = I_hsv(:,:,1);
S = I_hsv(:,:,2);
V = I_hsv(:,:,3);

% (a) 显示 原图 V 分量直方图
fig_a = figure('Name','(a) Original Intensity (Value) Histogram','Position',[50,200,1200,400]);
subplot(1,2,1); 
imshow(I_rgb);            
title('Original RGB Image');

subplot(1,2,2); 
imhist(V);                
title('Histogram of Original Intensity (V)');

% 保存 (a)
saveas(fig_a, fullfile(outputFolder, 'Fig_a_Original.png'));
close(fig_a);

% (b) 对 V 分量做直方图均衡化，不改 H 和 S
V_eq = histeq(V);  % 对亮度分量做直方图均衡化
I_hsv_eq = cat(3, H, S, V_eq);
I_rgb_eq = hsv2rgb(I_hsv_eq);

% 显示均衡化结果及直方图
fig_b = figure('Name','(b) Histogram Equalization on V','Position',[100,200,1200,400]);
subplot(1,2,1); 
imshow(I_rgb_eq);         
title('Histogram Equalized (V only)');

subplot(1,2,2); 
imhist(V_eq);             
title('Histogram of Equalized V');

saveas(fig_b, fullfile(outputFolder, 'Fig_b_EqualizedV.png'));
close(fig_b);

% 对比 (b) 的结果与原图
fig_b_cmp = figure('Name','(b) Compare Original and Equalized','Position',[150,250,1200,400]);
subplot(1,2,1); 
imshow(I_rgb);    
title('Original RGB');

subplot(1,2,2); 
imshow(I_rgb_eq); 
title('Equalized (V)');

saveas(fig_b_cmp, fullfile(outputFolder, 'Fig_b_Compare.png'));
close(fig_b_cmp);

% (c) 在 (b) 的基础上，“增加饱和度 S”
% 适度提高 (比如 +30%)，并截断到最大值 1
S_boost = 1.3 * S;
S_boost(S_boost > 1) = 1;

% 保持 V_eq 不变，组合新的 HSV
I_hsv_boosted = cat(3, H, S_boost, V_eq);
I_rgb_boosted = hsv2rgb(I_hsv_boosted);

% 显示结果
fig_c = figure('Name','(c) Increased Saturation','Position',[200,300,1200,400]);
subplot(1,2,1); 
imshow(I_rgb_boosted);
title('Equalized(V) + Boosted(S)');

subplot(1,2,2); 
imhist(I_hsv_boosted(:,:,3));
title('Histogram of V after Saturation Boost (same as eq)');

saveas(fig_c, fullfile(outputFolder, 'Fig_c_BoostedS.png'));
close(fig_c);

% 与 (b) 只做 V_eq 的结果进行对比
fig_c_cmp = figure('Name','(c) Compare with (b)','Position',[250,350,1200,400]);
subplot(1,2,1); 
imshow(I_rgb_eq);      
title('From (b) : V_{eq}');

subplot(1,2,2); 
imshow(I_rgb_boosted);
title('From (c) : V_{eq} + S_{boost}');

saveas(fig_c_cmp, fullfile(outputFolder, 'Fig_c_Compare.png'));
close(fig_c_cmp);

% (d) 显示 (c) 最终图像亮度分量直方图，并与 (a) 对比
V_after_c = I_hsv_boosted(:,:,3);  % 仍然是 V_eq

fig_d = figure('Name','(d) Compare final V histogram vs. original','Position',[300,400,1200,400]);
subplot(1,2,1); 
imhist(V_after_c);   
title('Histogram of final image (V)');

subplot(1,2,2); 
imhist(V);          
title('Histogram of original image (V)');

saveas(fig_d, fullfile(outputFolder, 'Fig_d_FinalHistogram.png'));
