function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:                d*N
%           - Y:                K*N
%           - GDparams:         object
%           - W:                K*d
%           - b:                K*1
%           - lambda:           1*1
%           - decay:            1*1
% OUTPUT    - Wstar:            K*d
%           - bstar:            K*1

n_batch = GDparams.n_batch;
eta = GDparams.eta;
N = size(X, 2);

for j = 1 : N/n_batch
    % generate mini-batches by running through images sequentially
    j_start = (j - 1)*n_batch + 1;
    j_end = j*n_batch;
    inds = j_start : j_end;
    Xbatch = X(:, inds);
    Ybatch = Y(:, inds);
    Xbatch = Xbatch + 0.08*randn(size(Xbatch));
    
    % compute gradients for each mini-batch
    P = EvaluateClassifier(Xbatch, W, b);
    [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
    
    % update weights and bias
    W = W - eta*grad_W;
    b = b - eta*grad_b;
end

Wstar = W;
bstar = b;

end