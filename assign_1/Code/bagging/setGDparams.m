function GDparams = setGDparams(n_batch, eta, n_epochs)
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