X = [1 3 4; 3 5 2; 4 4 2];
Y = [4 5 1; 2 8 5; 2 6 1];
% Y = [1 3 4; 3 5 2; 4 4 5];

dist = 0;

for i = 1 : 1 : length(X) * length(X)
    dist = dist + (X(i) - Y(i))^2;
end

dist = sqrt(dist)