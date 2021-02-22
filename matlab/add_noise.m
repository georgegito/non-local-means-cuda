clc; clear;

I = imread('../data/images/lena.jpg');
J = imnoise (I, 'gaussian');
figure();
imshow(I);
figure();
imshow(J);

normJ = mat2gray(J)

figure();
imshow(normJ);

save('lena.mat', 'normJ');
data=load('lena.mat');
field=fieldnames(data);
dlmwrite('lena.txt', data.(field{1}));