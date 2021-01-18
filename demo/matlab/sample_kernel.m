%% SCRIPT: SAMPLE_KERNEL
%
% Sample usage of GPU kernel through MATLAB
%
% DEPENDENCIES
%
%  sampleAddKernel.cu
%
  
  clear variables;

  %% PARAMETERS
  
  threadsPerBlock = [32 32];
  m = 1000;
  n = 1000;

  %% (BEGIN)

  fprintf('...begin %s...\n',mfilename);  
  
  %% KERNEL
  
  k = parallel.gpu.CUDAKernel( '../cuda/sampleAddKernel.ptx', ...
                               '../cuda/sampleAddKernel.cu');
  
  numberOfBlocks  = ceil( [m n] ./ threadsPerBlock );
  
  k.ThreadBlockSize = threadsPerBlock;
  k.GridSize        = numberOfBlocks;
  
  %% DATA
  
  A = rand([m n], 'gpuArray');
  B = zeros([m n], 'gpuArray');
  
  tic;
  B = gather( feval(k, A, B, m, n) );
  toc
  
  %% SANITY CHECK
  
  fprintf('Error: %e\n', norm( B - (A+1), 'fro' ) );
  
  %% (END)

  fprintf('...end %s...\n',mfilename);


%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION
%
%   0.1 - December 28, 2016
%
% CHANGELOG
%
%   0.1 (Dec 28, 2016) - Dimitris
%       * initial implementation
%
% ------------------------------------------------------------
