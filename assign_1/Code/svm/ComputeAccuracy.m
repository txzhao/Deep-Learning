function acc = ComputeAccuracy(X, y, W)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:            (d+1)*N
%           - y:            1*N
%           - W:            K*(d+1)
% OUTPUT    - acc:          1*1

s = EvaluateClassifier(X, W);
[~, k_star] = max(s);
acc = length(find(y - k_star == 0))/length(y);

end
