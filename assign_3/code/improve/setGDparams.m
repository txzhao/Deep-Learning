function GDparams = setGDparams(n_batch, eta, n_epochs)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - n_batch:          1*1
%           - eta:              1*1
%           - n_epochs:         1*1
% OUTPUT    - GDparams:         object

if nargin > 0
    GDparams.n_batch = n_batch;
    GDparams.eta = eta;
    GDparams.n_epochs = n_epochs;
end

end