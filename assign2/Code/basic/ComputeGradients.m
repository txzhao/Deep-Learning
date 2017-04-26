function [grad_W, grad_b] = ComputeGradients(X, Y, P, h, W, b, lambda)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:                d*N
%           - Y:                K*N
%           - P:                K*N
%           - W:                K*d
%           - lambda:           1*1
% OUTPUT    - grad_W:           K*d
%           - grad_b:           K*1

W1 = W{1};
W2 = W{2};
b1 = cell2mat(b(1));
b2 = cell2mat(b(2));
grad_W1 = zeros(size(W1));
grad_W2 = zeros(size(W2));
grad_b1 = zeros(size(b1));
grad_b2 = zeros(size(b2));

for i = 1 : size(X, 2)
    Pi = P(:, i);
    hi = h(:, i);
    Yi = Y(:, i);
    Xi = X(:, i);
    g = -Yi'*(diag(Pi) - Pi*Pi')/(Yi'*Pi);
    grad_b2 = grad_b2 + g';
    grad_W2 = grad_W2 + g'*hi';
    g = g*W2;
    hi(find(hi > 0)) = 1;
    g = g*diag(hi);
    grad_b1 = grad_b1 + g';
    grad_W1 = grad_W1 + g'*Xi';   
end

grad_W1 = 2*lambda*W1 + grad_W1/size(X, 2);
grad_W2 = 2*lambda*W2 + grad_W2/size(X, 2);
grad_b1 = grad_b1/size(X, 2);
grad_b2 = grad_b2/size(X, 2);
grad_W = {grad_W1, grad_W2}; 
grad_b = {grad_b1, grad_b2};

end
