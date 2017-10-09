function P = EvaluateClassifier(h, W, b)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:        d*N
%           - W:        K*d
%           - b:        K*1
% OUTPUT    - P:        K*N

W = W{end};
b = b{end};
X = h{end};
b = repmat(b, 1, size(X, 2));

s = W*X + b;
denorm = repmat(sum(exp(s), 1), size(s, 1), 1);
P = exp(s)./denorm;

end