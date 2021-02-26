save('house.mat', 'I');
save('noisy_house.mat', 'J');

data=load('house.mat');
field=fieldnames(data);
dlmwrite('house.txt', data.(field{1}));

data=load('noisy_house.mat');
field=fieldnames(data);
dlmwrite('noisy_house.txt', data.(field{1}));