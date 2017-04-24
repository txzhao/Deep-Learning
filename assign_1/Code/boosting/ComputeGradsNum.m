function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:                d*N
%           - Y:                K*N
%           - W:                K*d
%           - b:                K*1
%           - lambda:           1*1
%           - h:                1*1
% OUTPUT    - grad_W:           K*d
%           - grad_b:           K*1

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

c = ComputeCost(X, Y, W, b, lambda);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c) / h;
end

for i=1:numel(W)   
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c) / h;
end
