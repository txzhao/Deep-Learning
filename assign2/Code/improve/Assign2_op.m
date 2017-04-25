clear;
p = genpath('\assign1');
addpath(p);

% read in training, validation and test data
[Xtr1, Ytr1, ytr1] = LoadBatch('data_batch_1.mat');
[Xtr2, Ytr2, ytr2] = LoadBatch('data_batch_2.mat');
[Xtr3, Ytr3, ytr3] = LoadBatch('data_batch_3.mat');
[Xtr4, Ytr4, ytr4] = LoadBatch('data_batch_4.mat');
[Xtr5, Ytr5, ytr5] = LoadBatch('data_batch_5.mat');
[Xte, Yte, yte] = LoadBatch('test_batch.mat');

% regroup training data and validation data
Xtr = Xtr1;
Ytr = Ytr1;
ytr = ytr1;
% Xtr = Xtr(:, 1:100);
% Ytr = Ytr(:, 1:100);
% ytr = ytr(:, 1:100);
Xva = Xtr2(:, 1:1000);
Yva = Ytr2(:, 1:1000);
yva = ytr2(:, 1:1000);

% pre-processing
mean_X = mean(Xtr, 2);
Xtr = Xtr - repmat(mean_X, [1, size(Xtr, 2)]);
Xva = Xva - repmat(mean_X, [1, size(Xva, 2)]);
Xte = Xte - repmat(mean_X, [1, size(Xte, 2)]);

% initialization
m = 50;
n_epochs = 20;
n_batch = 100;
Eta = [];
Lambda = [];
acc_va = [];

%% random search 
% n_pairs = 50;
% eta_max = 0.3;
% eta_min = 0.01;
% lambda_max = 0.1;
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
%     [W, b, Jtr, Jva, flag] = main(Xtr, Ytr, Xva, Yva, GDparams, lambda, m);
%     toc
%     if flag == 0
%         Eta = [Eta, eta];
%         Lambda = [Lambda, lambda];
%         acc_va = [acc_va, ComputeAccuracy(Xva, yva, W, b)];
%     end
% end

%% evaluate networks
n_trial = 5;
n_bootstrap = 10000;
eta = 0.0622;
lambda = 1.6424e-6;
GDparams = setGDparams(n_batch, eta, n_epochs);
acc_n = zeros(1, n_trial);
P_n = zeros(size(Ytr, 1), n_bootstrap, n_trial);
decay = 0.95;

% training with the best hyper-parameters (Bagging)
for n = 1 : n_trial
    % generate bootstrap replicate
    idx = randperm(size(Xtr, 2), n_bootstrap);
    n_Xtr = Xtr(:, idx);
    n_Ytr = Ytr(:, idx);
    
    tic
    [W, b, Jtr, Jva, flag] = main(n_Xtr, n_Ytr, Xva, Yva, GDparams, lambda, m);
    toc
    
    [acc_te, P] = ComputeAccuracy(Xte, yte, W, b);
    acc_n(n) = acc_te;
    P_n(;, ;, n) = P;
    GDparams.eta = decay*GDparams.eta;
    
end

% evaluate accuracy of the final classifer (Bagging)
acc_n = acc_n/sum(acc_n);
P = zeros(size(P_n(:, :, 1)));
for n = 1 : n_trial
    P = P + P_n(:, :, n)*acc_n(n);
end
acc_te = easyaccuracy(P, yte);
disp(['test accuracy:' num2str(acc_te*100) '%'])

% % print accuracy of the network
% acc_tr = ComputeAccuracy(Xtr, ytr, W, b);
% disp(['training accuracy:' num2str(acc_tr*100) '%'])
% acc_te = ComputeAccuracy(Xte, yte, W, b);
% disp(['test accuracy:' num2str(acc_te*100) '%'])
% 
% % plot cost score
% figure()
% plot(1 : GDparams.n_epochs, Jtr, 'r')
% hold on
% plot(1 : GDparams.n_epochs, Jva, 'b')
% hold off
% xlabel('epoch');
% ylabel('loss');
% legend('training loss', 'validation loss');

% visualize weight matrix as class template images
% for i = 1 : K
% im = reshape(W(i, :), 32, 32, 3);
% s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
% s_im{i} = permute(s_im{i}, [2, 1, 3]);
% end
% figure()
% montage(s_im, 'size', [1, K])

