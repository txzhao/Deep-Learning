function [grad_W, grad_b] = ComputeGradients(X, Y, P, h, S, W, lambda, k, mu, v)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:                d*N
%           - Y:                K*N
%           - P:                K*N
%           - W:                K*d
%           - lambda:           1*1
% OUTPUT    - grad_W:           K*d
%           - grad_b:           K*1

grad_Wk = zeros(size(Y, 1), size(h{end}, 1));
grad_bk = zeros(size(Y, 1), 1);
g_prev = zeros(size(X, 2), size(h{end}, 1));
eps = 0.001;

for i = 1 : size(X, 2)
    Pi = P(:, i);
    hi = h{end}(:, i);
    Yi = Y(:, i);
    g = -Yi'*(diag(Pi) - Pi*Pi')/(Yi'*Pi);
    grad_bk = grad_bk + g';
    grad_Wk = grad_Wk + g'*hi';
    
    g = g*W{end};
    hi(find(hi > 0)) = 1;
%    hi(find(hi < 0)) = 0;
    hi(find(hi < 0)) = 0.01;            % leaky ReLu
    g_prev(i, :) = g*diag(hi);  
end

grad_W{k} = 2*lambda*W{end} + grad_Wk/size(X, 2);
grad_b{k} = grad_bk/size(X, 2);

for l = k - 1 : -1 : 1
    % batch normalization for backpass
    if nargin == 10
        g_prev = BN_backward(g_prev, mu{l}, v{l}, S{l}, eps);
    end

    grad_b{l} = mean(g_prev)';
    if l == 1
        grad_W{l} = g_prev'*X';
    else
        grad_W{l} = g_prev'*h{l - 1}';
    end
    grad_W{l} = grad_W{l}/size(X, 2) + 2*lambda*W{l};
    if l > 1
        g_prev = g_prev*W{l};
        H = h{l - 1};
        H(find(H > 0)) = 1;
%        H(find(H < 0)) = 0;
        H(find(H < 0)) = 0.01;          % leaky ReLu
        g_prev = g_prev.*H';            % equvalent to gi*diag(Ind(si>0))
    end    
end

end
