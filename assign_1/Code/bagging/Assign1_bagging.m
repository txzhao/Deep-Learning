clear;
addpath H:\assign1\Datasets\cifar-10-batches-mat\;

% read in training, validation and test data
[Xtr1, Ytr1, ytr1] = LoadBatch('data_batch_1.mat');
[Xtr2, Ytr2, ytr2] = LoadBatch('data_batch_2.mat');
[Xtr3, Ytr3, ytr3] = LoadBatch('data_batch_3.mat');
[Xtr4, Ytr4, ytr4] = LoadBatch('data_batch_4.mat');
[Xtr5, Ytr5, ytr5] = LoadBatch('data_batch_5.mat');
[Xte, Yte, yte] = LoadBatch('test_batch.mat');
Xtr = [Xtr1, Xtr2, Xtr3, Xtr4, Xtr5];
Ytr = [Ytr1, Ytr2, Ytr3, Ytr4, Ytr5];
ytr = [ytr1, ytr2, ytr3, ytr4, ytr5];
Xva = Xtr2(:, 1:1000);
Yva = Ytr2(:, 1:1000);
yva = ytr2(:, 1:1000);

% initialization
lambda = 0;
decay = 0.97;
n_trial = 5;
n_bootstrap = 10000;
GDparams = setGDparams(100, 0.01, 100);
acc_n = zeros(1, n_trial);
P_n = zeros(size(Ytr, 1), n_bootstrap, n_trial);

for n = 1 : n_trial
    % generate bootstrap replicate
    idx = randperm(size(Xtr, 2), n_bootstrap);
    n_Xtr = Xtr(:, idx);
    n_Ytr = Ytr(:, idx);
    
    % iterate mini-batch gradient descent algorithm
    [W, b, Jtr, Jva] = main(n_Xtr, n_Ytr, Xva, Yva, GDparams, lambda);
    
    % evaluate base classifier
    [acc_te, P] = ComputeAccuracy(Xte, yte, W, b);
    acc_n(n) = acc_te;
    P_n(:, :, n) = P;
    GDparams.eta = decay*GDparams.eta;
    
    % plot cost score
    figure()
    plot(1 : GDparams.n_epochs, Jtr, 'r')
    hold on
    plot(1 : GDparams.n_epochs, Jva, 'b')
    hold off
    xlabel('epoch');
    ylabel('loss');
    legend('training loss', 'validation loss');
    title(['n_trial = ' num2str(n)]);
    
end

% evaluate accuracy of the final classifier
acc_n = acc_n/sum(acc_n);
P = zeros(size(P_n(:, :, 1)));
for n = 1 : n_trial
    P = P + P_n(:, :, n)*acc_n(n);
end
acc_te = easyaccuracy(P, yte);
disp(['test accuracy:' num2str(acc_te*100) '%'])

