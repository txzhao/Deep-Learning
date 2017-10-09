function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda, rho)
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
v_W = {zeros(size(W{1})), zeros(size(W{2}))};
v_b = {zeros(size(b{1})), zeros(size(b{2}))};

for j = 1 : N/n_batch
    % generate mini-batches by running through images sequentially
    j_start = (j - 1)*n_batch + 1;
    j_end = j*n_batch;
    inds = j_start : j_end;
    Xbatch = X(:, inds);
    Ybatch = Y(:, inds);
%     Xbatch = Xbatch + 0.08*randn(size(Xbatch));
    
    % compute gradients for each mini-batch
    h = intervalues(Xbatch, W, b);
    P = EvaluateClassifier(h, W, b);
    [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, h, W, b, lambda);
    
    % add momentum
    v_W{1} = rho*v_W{1} + eta*grad_W{1};
    v_W{2} = rho*v_W{2} + eta*grad_W{2};
    v_b{1} = rho*v_b{1} + eta*grad_b{1};
    v_b{2} = rho*v_b{2} + eta*grad_b{2};
    
    % update weights and bias
    W{1} = W{1} - v_W{1};
    W{2} = W{2} - v_W{2};
    b{1} = b{1} - v_b{1};
    b{2} = b{2} - v_b{2};
end

Wstar = W;
bstar = b;

end