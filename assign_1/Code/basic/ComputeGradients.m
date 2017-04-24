function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:                d*N
%           - Y:                K*N
%           - P:                K*N
%           - W:                K*d
%           - lambda:           1*1
% OUTPUT    - grad_W:           K*d
%           - grad_b:           K*1

grad_W = zeros(size(W));
grad_b = zeros(size(W, 1), 1);

for i = 1 : size(X, 2)
    Pi = P(:, i);
    Yi = Y(:, i);
    Xi = X(:, i);
    g = -Yi'*(diag(Pi) - Pi*Pi')/(Yi'*Pi);
    grad_b = grad_b + g';
    grad_W = grad_W + g'*Xi';
end

grad_b = grad_b/size(X, 2);
grad_W = 2*lambda*W + grad_W/size(X, 2);

end
