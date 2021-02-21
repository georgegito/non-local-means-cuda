clc; clear; close all;
Files=dir('../data/out/*');
for k=1:length(Files)
    if startsWith(Files(k).name,'.') || startsWith(Files(k).name,'e')
        continue
    end
    path = "../data/out/" + Files(k).name;
    image = dlmread(path);
    
    idx = 0;
    if contains(path, 'cuda')
        cuda = 'CUDA ';
        idx = 1;
    else
        cuda = ''; 
    end
    
    if contains(path, 'filtered')
        name = [cuda 'Filtered image'];
        tmp = split(path, '_');
        tmptmp = split(tmp(5 + idx), '.txt');
        patchSize = tmp(3 + idx);
        patchSigma = tmp(4 + idx);
        filterSigma = tmptmp(1 + idx);
    else 
        name = [cuda 'Residual'];
        tmp = split(path, '_');
        tmptmp = split(tmp(4 + idx), '.txt');
        patchSize = tmp(2 + idx);
        patchSigma = tmp(3 + idx);
        filterSigma = tmptmp(1 + idx);    
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