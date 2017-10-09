function [W, b, Jtr, Jva, flag] = main(Xtr, Ytr, Xva, Yva, GDparams, lambda, m)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds, T as the number of epochs
% INPUT     - Xtr:          d*N
%           - Ytr:          K*N
%           - GDparams:     object
%           - lambda:       1*1
% OUTPUT    - W:            K*d
%           - b:            K*1
%           - Jtr:          1*T

% initialize W and b
d = size(Xtr, 1);
K = size(Ytr, 1);
std = 0.001;
[W, b] = init_para(m, d, K, std);


% check if analytical gradient is correct
% batch_size = 100;
% delta = 1e-6;
% [gradcheck_W, gradcheck_b] = gradCheck(batch_size, Xtr, Ytr, lambda, W, b, delta);

% perform the mini-batch gradient descent algorithm
Jtr = zeros(1, GDparams.n_epochs);
Jva = zeros(1, GDparams.n_epochs);
decay_rate = 0.8;
rho = 0.9;
flag = 0;
for i = 1 : GDparams.n_epochs
%     Jtr(i) = ComputeCost(Xtr, Ytr, W, b, lambda);
%     Jva(i) = ComputeCost(Xva, Yva, W, b, lambda);
%     if Jtr(i) > 3*Jtr(1)
%         flag = 1;
%         break;
%     end
    [W, b] = MiniBatchGD(Xtr, Ytr, GDparams, W, b, lambda, rho);

%	anneal learning rate
%     if mod(i, 4) == 0
%         GDparams.eta = 0.4*GDparams.eta;
%     end
    GDparams.eta = decay_rate*GDparams.eta;
end

end