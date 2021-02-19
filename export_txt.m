load('ws.mat');

save('noised_house.mat', 'J');

data=load('noised_house.mat');
field=fieldnames(data);
dlmwrite('noised_house.txt', data.(field{1}));