function [W, b, Jtr, Jva, flag, mu_av, v_av] = main(Xtr, Ytr, Xva, Yva, GDparams, lambda, k, m)
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
M = [d, m, K];
std = 0.001;
[W, b] = init_para(M, std);

% check if analytical gradient is correct
% batch_size = 1;
% delta = 1e-6;
% gradCheck(batch_size, Xtr, Ytr, lambda, W, b, delta, k)

% perform the mini-batch gradient descent algorithm
Jtr = zeros(1, GDparams.n_epochs);
Jva = zeros(1, GDparams.n_epochs);
decay_rate = 0.8;
rho = 0.9;
alpha = 0.99;
flag = 0;

for i = 1 : GDparams.n_epochs
    [W, b, mu_av, v_av] = MiniBatchGD(Xtr, Ytr, GDparams, W, b, lambda, rho, alpha, k);
    Jtr(i) = ComputeCost(Xtr, Ytr, W, b, lambda, k, mu_av, v_av);
    Jva(i) = ComputeCost(Xva, Yva, W, b, lambda, k, mu_av, v_av);
%     if Jtr(i) > 3*Jtr(1)
%         flag = 1;
%         break;
%     end
    GDparams.eta = decay_rate*GDparams.eta;
end

end
