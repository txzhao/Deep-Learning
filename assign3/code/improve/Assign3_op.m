clear;
addpath /Users/MMGF2/Desktop/deep_learning_in_data_science/Datasets/cifar-10-batches-mat;

% read in training, validation and test data
[Xtr1, Ytr1, ytr1] = LoadBatch('data_batch_1.mat');
[Xtr2, Ytr2, ytr2] = LoadBatch('data_batch_2.mat');
[Xtr3, Ytr3, ytr3] = LoadBatch('data_batch_3.mat');
[Xtr4, Ytr4, ytr4] = LoadBatch('data_batch_4.mat');
[Xtr5, Ytr5, ytr5] = LoadBatch('data_batch_5.mat');
[Xte, Yte, yte] = LoadBatch('test_batch.mat');

% regroup training data and validation data
Xtr = [Xtr1, Xtr2, Xtr3, Xtr4, Xtr5];
Ytr = [Ytr1, Ytr2, Ytr3, Ytr4, Ytr5];
ytr = [ytr1, ytr2, ytr3, ytr4, ytr5];
Xva = Xtr2;
Yva = Ytr2;
yva = ytr2;

% pre-processing
mean_X = mean(Xtr, 2);
Xtr = Xtr - repmat(mean_X, [1, size(Xtr, 2)]);
Xva = Xva - repmat(mean_X, [1, size(Xva, 2)]);
Xte = Xte - repmat(mean_X, [1, size(Xte, 2)]);

% initialization
k = 3;
m = [50 30];
n_epochs = 30;
n_batch = 100;
Eta = [];
Lambda = [];
acc_va = [];

%% random search 
% n_pairs = 50;
% eta_max = 0.9;
% eta_min = 0.028;
% lambda_max = 5.3e-3;
% lambda_min = 1e-7;
% 
% for i = 1 : n_pairs
%     % randomly generate eta and lambda
%     e = log10(eta_min) + (log10(eta_max) - log10(eta_min))*rand(1, 1); 
%     eta = 10^e;
%     e = log10(lambda_min) + (log10(lambda_max) - log10(lambda_min))*rand(1, 1);
%     lambda = 10^e;
%     
%     GDparams = setGDparams(n_batch, eta, n_epochs);
%     tic
%     [W, b, Jtr, Jva, flag, mu_av, v_av] = main(Xtr, Ytr, Xva, Yva, GDparams, lambda, k, m);
%     toc
%     
%     if flag == 0
%         Eta = [Eta, eta];
%         Lambda = [Lambda, lambda];
%         acc_va = [acc_va, ComputeAccuracy(Xva, yva, W, b, k, mu_av, v_av)];
%     end
%     disp(['i = ' num2str(i) ', test accuracy = ' num2str(acc_va(i)*100) '%'])
% end

%% train networks
n_trial = 6;
n_bootstrap = 10000;
Eta = [0.4138, 0.3850, 0.8745, 0.1499, 0.5264, 0.3595];
Lambda = [2.8193e-5, 1.3144e-04, 8.1984e-04, 7.2835e-04, 6.4889e-04, 7.7693e-06];
acc_n = zeros(1, n_trial);
P_n = zeros(size(Yte, 1), size(Yte, 2), n_trial);

% training with the best hyper-parameters
for n = 1 : n_trial
    % initialize parameters
    eta = Eta(n);
    lambda = Lambda(n);
    GDparams = setGDparams(n_batch, eta, n_epochs);
    
    % generate bootstrap replicate
    idx = randperm(size(Xtr, 2), n_bootstrap);
    n_Xtr = Xtr(:, idx);
    n_Ytr = Ytr(:, idx);

    tic
    [W, b, ~, ~, ~, mu_av, v_av] = main(Xtr, Ytr, Xva, Yva, GDparams, lambda, k, m);
    toc
    
    [~, P] = ComputeAccuracy(Xte, yte, W, b, k, mu_av, v_av);
    P_n(:, :, n) = P;
    
end


%% evaluate networks
% evaluate accuracy of the final classifer (Bagging)
P = zeros(size(P_n(:, :, 1)));
for n = 1 : n_trial
    P = P + P_n(:, :, n);
end
acc_te = easyaccuracy(P, yte);
disp(['test accuracy:' num2str(acc_te*100) '%'])


%% ===================  function involved  ==========================
function g = BN_backward(g, mu, v, S, eps)

g_vb = g*diag((v + eps).^(-1.5)).*(S - repmat(mu, 1, size(S, 2)))';
g_vb = 0.5*sum(g_vb);
g_mub = -sum(g*diag((v + eps).^(-0.5)));
g = g*diag((v + eps).^(-0.5)) + 2/size(S, 2)*repmat(g_vb,...
        size(S, 2), 1).*(S - repmat(mu, 1, size(S, 2)))' + ...
        repmat(g_mub, size(S, 2), 1)/size(S, 2);
    
end

function [sbar, mu, v] = BN_forward(s, eps, mu_av, v_av)
if nargin < 4
    mu = mean(s, 2);
    v = mean((s - repmat(mu, 1, size(s, 2))).^2, 2);
else
    mu = mu_av;
    v = v_av;
end
sbar = diag((v + eps).^(-0.5))*(s - repmat(mu, 1, size(s, 2)));

end

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

function [grad_W, grad_b] = ComputeGradients(X, Y, P, h, S, W, lambda, k, mu, v)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:                d*N
%           - Y:                K*N
%           - P:                K*N
%           - W:                K*d
%           - lambda:           1*1
% OUTPUT    - grad_W:           K*d
%           - grad_b:           K*1

grad_Wk = zeros(size(Y, 1), size(h{end}, 1));
grad_bk = zeros(size(Y, 1), 1);
g_prev = zeros(size(X, 2), size(h{end}, 1));
eps = 0.001;

for i = 1 : size(X, 2)
    Pi = P(:, i);
    hi = h{end}(:, i);
    Yi = Y(:, i);
    g = -Yi'*(diag(Pi) - Pi*Pi')/(Yi'*Pi);
    grad_bk = grad_bk + g';
    grad_Wk = grad_Wk + g'*hi';
    
    g = g*W{end};
    hi(find(hi > 0)) = 1;
%     hi(find(hi < 0)) = 0;
    hi(find(hi < 0)) = 0.01;        % leaky ReLu
    g_prev(i, :) = g*diag(hi);  
end

grad_W{k} = 2*lambda*W{end} + grad_Wk/size(X, 2);
grad_b{k} = grad_bk/size(X, 2);

for l = k - 1 : -1 : 1
    % batch normalization for backpass
    if nargin == 10
        g_prev = BN_backward(g_prev, mu{l}, v{l}, S{l}, eps);
    end

    grad_b{l} = mean(g_prev)';
    if l == 1
        grad_W{l} = g_prev'*X';
    else
        grad_W{l} = g_prev'*h{l - 1}';
    end
    grad_W{l} = grad_W{l}/size(X, 2) + 2*lambda*W{l};
    if l > 1
        g_prev = g_prev*W{l};
        H = h{l - 1};
        H(find(H > 0)) = 1;
%         H(find(H < 0)) = 0;
        H(find(H < 0)) = 0.01;      % leaky ReLu
        g_prev = g_prev.*H';        % equvalent to gi*diag(Ind(si>0))
    end    
end

end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda, length(b));
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda, length(b));
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda, length(W));
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda, length(W));
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end

function acc = easyaccuracy(P, y)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - P:            K*N
%           - y:            1*N
% OUTPUT    - acc:          1*1

[~, k_star] = max(P);
acc = length(find(y - k_star == 0))/length(y);
end

function P = EvaluateClassifier(h, W, b)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:        d*N
%           - W:        K*d
%           - b:        K*1
% OUTPUT    - P:        K*N

W = W{end};
b = b{end};
X = h{end};
b = repmat(b, 1, size(X, 2));

s = W*X + b;
denorm = repmat(sum(exp(s), 1), size(s, 1), 1);
P = exp(s)./denorm;

end

function gradCheck(batch_size, X, Y, lambda, W, b, delta, k)

% numerical gradients
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(X(:, 1 : batch_size), ...
    Y(:, 1 : batch_size), W, b, lambda, delta);

% analytical gradients
[h, S, mu, v] = intervalues(X(:, 1 : batch_size), W, b, k);
P = EvaluateClassifier(h, W, b);
[grad_W, grad_b] = ComputeGradients(X(:, 1 : batch_size), ...
    Y(:, 1 : batch_size), P, h, S, W, lambda, k, mu, v);

% relative error rate
eps = 0.001;
for i = 1 : length(W)
    gradcheck_bi = sum(abs(ngrad_b{i} - grad_b{i})/max(eps, sum(abs(ngrad_b{i}) + abs(grad_b{i}))));
    gradcheck_bm = max(abs(ngrad_b{i} - grad_b{i})/max(eps, sum(abs(ngrad_b{i}) + abs(grad_b{i}))));
    gradcheck_Wi = sum(sum(abs(ngrad_W{i} - grad_W{i})/max(eps, sum(sum(abs(ngrad_W{i}) + abs(grad_W{i}))))));
    gradcheck_Wm = max(max(abs(ngrad_W{i} - grad_W{i})/max(eps, sum(sum(abs(ngrad_W{i}) + abs(grad_W{i}))))));
    disp(['error of grad_W' num2str(i) ': ' num2str(gradcheck_Wi)])
    disp(['max error of grad_W' num2str(i) ': ' num2str(gradcheck_Wm)])
    disp(['error of grad_b' num2str(i) ': ' num2str(gradcheck_bi)])
    disp(['max error of grad_b' num2str(i) ': ' num2str(gradcheck_bm)])
end

end

function [W, b] = init_para(m, std)

for i = 1 : size(m, 2) - 1
    W{i} = std*rand(m(i + 1), m(i));
    b{i} = zeros(m(i + 1), 1);
end

end

function [h, S, mu, v] = intervalues(X, W, b, k, mu_av, v_av)

eps = 0.001;
for i = 1 : k - 1
    Wi = W{i};
    bi = b{i};
    bi = repmat(bi, 1, size(X, 2));
    s = Wi*X + bi;
    S{i} = s;
    
    % batch normalization
    if nargin < 6
        [sbar, mui, vi] = BN_forward(s, eps);
    else
        [sbar, mui, vi] = BN_forward(s, eps, mu_av{i}, v_av{i});
    end
    mu{i} = mui;
    v{i} = vi;

%     X = max(0, sbar);
    X = max(0.01*sbar, sbar);       % leaky ReLu
    h{i} = X;
end

end

function [X, Y, y] = LoadBatch(filename)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - filename:     string
% OUTPUT    - X:            d*N
%           - Y:            K*N
%           - y:            1*N

indata = load(filename);
X = double(indata.data')/255;   % convert image from 0-256 to 0-1
y = double(indata.labels') + 1;
Y = oneHot(y);                  % convert labels to one-hot representation

end

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
%     Jtr(i) = ComputeCost(Xtr, Ytr, W, b, lambda, k, mu_av, v_av);
%     Jva(i) = ComputeCost(Xva, Yva, W, b, lambda, k, mu_av, v_av);
%     if Jtr(i) > 3*Jtr(1)
%         flag = 1;
%         break;
%     end
    GDparams.eta = decay_rate*GDparams.eta;
end

end

function [Wstar, bstar, mu_av, v_av] = MiniBatchGD(X, Y, GDparams, W, b, lambda, rho, alpha, k)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:                d*N
%           - Y:                K*N
%           - GDparams:         object
%           - W:                K*d
%           - b:                K*1
%           - lambda:           1*1
%           - decay:            1*1
% OUTPUT    - Wstar:            K*d
%           - bstar:            K*1

n_batch = GDparams.n_batch;
eta = GDparams.eta;
N = size(X, 2);

for i = 1 : k
    v_W{i} = zeros(size(W{i}));
    v_b{i} = zeros(size(b{i}));
end

for j = 1 : N/n_batch
    % generate mini-batches by running through images sequentially
    j_start = (j - 1)*n_batch + 1;
    j_end = j*n_batch;
    inds = j_start : j_end;
    Xbatch = X(:, inds);
    Ybatch = Y(:, inds);
    Xbatch = Xbatch + 0.12*randn(size(Xbatch));         % add a random jitter
    
    % compute gradients for each mini-batch
    [h, S, mu, v] = intervalues(Xbatch, W, b, k);
    P = EvaluateClassifier(h, W, b);
    
    for i = 1 : k - 1
        if j == 1
            mu_av = mu;
            v_av = v;
        else
            mu_av{i} = alpha*mu_av{i} + (1 - alpha)*mu{i};
            v_av{i} = alpha*v_av{i} + (1 - alpha)*v{i};
        end
    end
    
    [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, h, S, W, lambda, k, mu, v);
    
    for i = 1 : k
        % add momentum
        v_W{i} = rho*v_W{i} + eta*grad_W{i};
        v_b{i} = rho*v_b{i} + eta*grad_b{i};
    
        % update weights and bias
        W{i} = W{i} - v_W{i};
        b{i} = b{i} - v_b{i};
    end
end

Wstar = W;
bstar = b;

end

function out = oneHot(label)
% DENOTE N as the number of images, K as the number of label kinds
% INPUT     - label:        1*N
% OUTPUT    - out:          K*N

K = length(unique(label));
N = length(label);
out = zeros(K, N);
for i = 1 : N
    out(label(i), i) = 1;
end

end

function GDparams = setGDparams(n_batch, eta, n_epochs)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - n_batch:          1*1
%           - eta:              1*1
%           - n_epochs:         1*1
% OUTPUT    - GDparams:         object

if nargin > 0
    GDparams.n_batch = n_batch;
    GDparams.eta = eta;
    GDparams.n_epochs = n_epochs;
end

end

