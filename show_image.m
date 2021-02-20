clc; clear;

image = dlmread('filtered_image.txt');
save('filtered_image.mat', 'image')
figure('Name', 'Filtered image');
imagesc(image); 
colormap gray;

image = dlmread('residual.txt');
save('residual.mat', 'image')
figure('Name', 'Residual');
imagesc(image); 
colormap gray;

