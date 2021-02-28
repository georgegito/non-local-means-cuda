clc; clear;

I = imread('../data/images/flower.jpg');
J = imnoise (I, 'gaussian');
figure();
imagesc(I);
title('original image'); colormap gray;
figure();
imagesc(J);
title('noisy image'); colormap gray;

normJ = mat2gray(J)

figure();
% imagesc(normJ);

save('flower.mat', 'normJ');
data=load('flower.mat');
field=fieldnames(data);
dlmwrite('flower.txt', data.(field{1}));