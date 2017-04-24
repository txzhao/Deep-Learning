function J = ComputeCost(X, Y, W, lambda, delta)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:            (d+1)*N
%           - Y:            K*N
%           - W:            K*(d+1)
%           - lambda:       1*1
%           - delta:        1*1
% OUTPUT    - J:            1*1

s = EvaluateClassifier(X, W);
sc = repmat(sum(s.*Y), size(s, 1), 1);
margin = s - sc + delta;
J1 = sum(margin(find(margin > 0))) - size(s, 2)*delta;
J1 = J1/size(s, 2);
J2 = lambda*sum(sum(W.^2));
J = J1 + J2;

end