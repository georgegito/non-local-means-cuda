clc; clear; close all;
Files=dir('../data/out/*');
for k=1:length(Files)
    if startsWith(Files(k).name,'.')
        continue
    end
    path = "../data/out/" + Files(k).name;
    image = dlmread(path);
    if contains(path, 'filtered')
        name = 'Filtered image';
        tmp = split(path, '_');
        tmptmp = split(tmp(5), '.txt');
        patchSize = tmp(3);
        patchSigma = tmp(4);
        filterSigma = tmptmp(1);
    else 
        name = 'Residual';
        tmp = split(path, '_');
        tmptmp = split(tmp(4), '.txt');
        patchSize = tmp(2);
        patchSigma = tmp(3);
        filterSigma = tmptmp(1);    
    end
    figure('Name', name);
    imagesc(image); 
    hold on;
    patchsize = "Patch Size = " + patchSize;
    patchsigma = "Patch Sigma = " + patchSigma;
    filtersigma = "Filter Sigma = " + filterSigma;
    title({patchsize, patchsigma, filtersigma})
    colormap gray;
    drawnow
end