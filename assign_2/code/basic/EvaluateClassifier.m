function P = EvaluateClassifier(h, W, b)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:        d*N
%           - W:        K*d
%           - b:        K*1
% OUTPUT    - P:        K*N

W2 = cell2mat(W(2));
b2 = cell2mat(b(2));
b2 = repmat(b2, 1, size(h, 2));
s = W2*h + b2;
denorm = repmat(sum(exp(s), 1), size(W2, 1), 1);
P = exp(s)./denorm;

end