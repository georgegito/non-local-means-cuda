save('house.mat', 'I');
save('noised_house.mat', 'J');

data=load('house.mat');
field=fieldnames(data);
dlmwrite('house.txt', data.(field{1}));

data=load('noised_house.mat');
field=fieldnames(data);
dlmwrite('noised_house.txt', data.(field{1}));