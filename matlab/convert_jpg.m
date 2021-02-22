image=imread('../data/images/lena.jpg');
im2double(image);
save('image.mat', 'image');
imshow(image);