image=imread('../data/images/flower.jpg');
im2double(image);
save('image.mat', 'image');
imshow(image);