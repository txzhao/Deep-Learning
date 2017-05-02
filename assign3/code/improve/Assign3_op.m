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

% training with the best hyper-parameters (Bagging)
for n = 1 : n_trial
    % initialize parameters
    eta = Eta(n);
    lambda = Lambda(n);
    GDparams = setGDparams(n_batch, eta, n_epochs);
    
    % randomly generate bootstrap replicate
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
acc_te = easyaccuracy(P, yte);              % majority vote
disp(['test accuracy:' num2str(acc_te*100) '%'])


