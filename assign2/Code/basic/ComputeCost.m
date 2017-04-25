function J = ComputeCost(X, Y, W, b, lambda)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:            d*N
%           - Y:            K*N
%           - W:            K*d
%           - b:            K*1
%           - lambda:       1*1
% OUTPUT    - J:            1*1

W1 = W{1};
W2 = W{2};

h = intervalues(X, W, b);
P = EvaluateClassifier(h, W, b);
J1 = sum(diag(-log(Y'*P)))/size(X, 2);
J2 = lambda*(sum(sum(W1.^2)) + sum(sum(W2.^2)));
J = J1 + J2;

end