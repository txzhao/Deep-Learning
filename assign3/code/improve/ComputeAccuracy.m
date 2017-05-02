function [acc, k_star] = ComputeAccuracy(X, y, W, b, k, mu_av, v_av)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:            d*N
%           - y:            1*N
%           - W:            K*d
%           - b:            K*1
% OUTPUT    - acc:          1*1
%           - k_star:       K*N

if nargin < 7
    h = intervalues(X, W, b, k);
else
    h = intervalues(X, W, b, k, mu_av, v_av);
end
P = EvaluateClassifier(h, W, b);
[~, k_star] = max(P);
acc = length(find(y - k_star == 0))/length(y);
k_star = oneHot(k_star);

end
