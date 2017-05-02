function J = ComputeCost(X, Y, W, b, lambda, k, mu_av, v_av)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:            d*N
%           - Y:            K*N
%           - W:            K*d
%           - b:            K*1
%           - lambda:       1*1
% OUTPUT    - J:            1*1

if nargin < 8
    h = intervalues(X, W, b, k);
else
    h = intervalues(X, W, b, k, mu_av, v_av);
end
P = EvaluateClassifier(h, W, b);
J1 = sum(diag(-log(Y'*P)))/size(X, 2);

J2 = 0;
for i = 1 : length(W)
    temp = W{i}.^2;
    J2 = J2 + lambda*sum(temp(:));
end

J = J1 + J2;

end