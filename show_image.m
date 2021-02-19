image = readmatrix('filtered_image.txt');
save('filtered_image.mat', 'Image')
figure('Name', 'Filtered image');
imagesc(image); 
colormap gray;