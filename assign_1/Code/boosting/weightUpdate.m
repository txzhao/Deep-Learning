function [alpha, wCur] = weightUpdate(X, y, W, b, wCur)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:            d*N
%           - y:            1*N
%           - W:            K*d
%           - b:            K*1
%           - wCur:         1*N
% OUTPUT    - alpha:        1*1
%           - wCur:         1*N

P = EvaluateClassifier(X, W, b);
[~, k_star] = max(P);
a = zeros(1, size(X, 2));
a(find(y - k_star == 0)) = 1;
error = sum(a.*wCur);
alpha = 0.5*(log(1 - error) - log(error));
wCur(find(a == 1)) = wCur(find(a == 1))*exp(2*alpha);
wCur = wCur*exp(-alpha);
wCur = wCur/sum(wCur);

end
