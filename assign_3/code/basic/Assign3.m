clear;
addpath /Users/MMGF2/Desktop/deep_learning_in_data_science/Datasets/cifar-10-batches-mat;

% read in training, validation and test data
[Xtr1, Ytr1, ytr1] = LoadBatch('data_batch_1.mat');
[Xtr2, Ytr2, ytr2] = LoadBatch('data_batch_2.mat');
% [Xtr3, Ytr3, ytr3] = LoadBatch('data_batch_3.mat');
% [Xtr4, Ytr4, ytr4] = LoadBatch('data_batch_4.mat');
% [Xtr5, Ytr5, ytr5] = LoadBatch('data_batch_5.mat');
[Xte, Yte, yte] = LoadBatch('test_batch.mat');

% regroup training data and validation data
Xtr = Xtr1;
Ytr = Ytr1;
ytr = ytr1;
% Xtr = Xtr(:, 1:100);
% Ytr = Ytr(:, 1:100);
% ytr = ytr(:, 1:100);
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
n_epochs = 50;
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
%     %randomly generate eta and lambda
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
eta = 0.4380;
lambda = 1.0907e-4;
GDparams = setGDparams(n_batch, eta, n_epochs);

% training with the best hyper-parameters
tic
[W, b, Jtr, Jva, ~, mu_av, v_av] = main(Xtr, Ytr, Xva, Yva, GDparams, lambda, k, m);
toc

%% evaluate networks
% print accuracy of the network
acc_tr = ComputeAccuracy(Xtr, ytr, W, b, k, mu_av, v_av);
disp(['training accuracy:' num2str(acc_tr*100) '%'])
acc_te = ComputeAccuracy(Xte, yte, W, b, k, mu_av, v_av);
disp(['test accuracy:' num2str(acc_te*100) '%'])

% plot cost score
figure()
plot(1 : GDparams.n_epochs, Jtr, 'r')
hold on
plot(1 : GDparams.n_epochs, Jva, 'b')
hold off
xlabel('epoch');
ylabel('loss');
legend('training loss', 'validation loss');

