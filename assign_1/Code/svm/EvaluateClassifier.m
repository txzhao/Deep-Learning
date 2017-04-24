function s = EvaluateClassifier(X, W)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:        (d+1)*N
%           - W:        K*(d+1)
% OUTPUT    - s:        K*N

s = W*X;

end