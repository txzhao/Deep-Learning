function [W, b, Jtr] = main(Xtr, Ytr, GDparams, lambda)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds, T as the number of epochs
% INPUT     - Xtr:          d*N
%           - Ytr:          K*N
%           - GDparams:     object
%           - lambda:       1*1
% OUTPUT    - W:            K*d
%           - b:            K*1
%           - Jtr:          1*T

% initialize W and b
mean = 0;
std = 0.01;
K = size(Ytr, 1);
d = size(Xtr, 1);
W = mean + randn(K, d)*std;
b = mean + randn(K, 1)*std;


% check if analytical gradient is correct
% batch_size = 50;
% [ngrad_b, ngrad_W] = ComputeGradsNumSlow(Xtr(:, 1 : batch_size), ...
%     Ytr(:, 1 : batch_size), W, b, lambda, 1e-6);
% P = EvaluateClassifier(Xtr(:, 1 : batch_size), W, b);
% [grad_W, grad_b] = ComputeGradients(Xtr(:, 1 : batch_size), ...
%     Ytr(:, 1 : batch_size), P, W, lambda);
% gradcheck_b = max(abs(ngrad_b - grad_b)./max(0, abs(ngrad_b) + abs(grad_b)))
% gradcheck_W = max(max(abs(ngrad_W - grad_W)./max(0, abs(ngrad_W) + abs(grad_W))))

% perform the mini-batch gradient descent algorithm
Jtr = zeros(1, GDparams.n_epochs);
% Jva = zeros(1, GDparams.n_epochs);
for i = 1 : GDparams.n_epochs
    Jtr(i) = ComputeCost(Xtr, Ytr, W, b, lambda);
%     Jva(i) = ComputeCost(Xva, Yva, W, b, lambda);
%     if i ~= 1 && Jva(i - 1) - Jva(i) < 0
%         break;
%     end
    [W, b] = MiniBatchGD(Xtr, Ytr, GDparams, W, b, lambda);
end

end