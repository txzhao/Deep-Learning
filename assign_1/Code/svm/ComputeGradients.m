function grad_W = ComputeGradients(X, Y, s, W, delta, lambda)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:                (d+1)*N
%           - Y:                K*N
%           - s:                K*N
%           - W:                K*(d+1)
%           - lambda:           1*1
%           - delta:            1*1
% OUTPUT    - grad_W:           K*(d+1)

grad_W = zeros(size(W));
sc = repmat(sum(s.*Y), size(s, 1), 1);
margin = s - sc + delta;
flag = zeros(size(s));
flag(find(margin > 0)) = 1;
flag(find(Y == 1)) = -1;

for i = 1 : size(X, 2)
    Xi = X(:, i);
    fi = flag(:, i);
    gi = repmat(Xi', size(W, 1), 1);
    gi(find(fi == 0), :) = 0;
    gi(find(fi == -1), :) = -length(find(fi == 1))*gi(find(fi == -1), :);
    grad_W = grad_W + gi;
end

grad_W = 2*lambda*W + grad_W/size(X, 2);

end
