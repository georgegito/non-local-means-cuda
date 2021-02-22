clc; clear;

I = imread('../data/images/flower.jpg');
J = imnoise (I, 'gaussian');
figure();
imshow(I);
figure();
imshow(J);

normJ = mat2gray(J)

figure();
imshow(normJ);

save('flower.mat', 'normJ');
data=load('flower.mat');
field=fieldnames(data);
dlmwrite('flower.txt', data.(field{1}));