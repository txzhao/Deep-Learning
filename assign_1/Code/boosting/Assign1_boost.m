clear;
addpath H:\assign1\Datasets\cifar-10-batches-mat\;

% read in training, validation and test data
[Xtr1, Ytr1, ytr1] = LoadBatch('data_batch_1.mat');
[Xtr2, Ytr2, ytr2] = LoadBatch('data_batch_2.mat');
[Xtr3, Ytr3, ytr3] = LoadBatch('data_batch_3.mat');
[Xtr4, Ytr4, ytr4] = LoadBatch('data_batch_4.mat');
[Xtr5, Ytr5, ytr5] = LoadBatch('data_batch_5.mat');
[Xva, Yva, yva] = LoadBatch('data_batch_2.mat');
[Xte, Yte, yte] = LoadBatch('test_batch.mat');
Xtr = Xtr1;
Ytr = Ytr1;
ytr = ytr1;
Xva = Xva(:, 1:1000);
Yva = Yva(:, 1:1000);
yva = yva(:, 1:1000);

% initialization
lambda = 0;
decay = 0.9;
GDparams = setGDparams(100, 0.01, 40);
n_trial = 10;
Wf = zeros(size(Ytr, 1), size(Xtr, 1), n_trial);
bf = zeros(size(Ytr, 1), n_trial);
wCur = 1/size(Xtr, 2)*ones(1, size(Xtr, 2));
alphaf = zeros(1, n_trial);

% generate base classifiers
for n = 1 : n_trial
    [W, b] = main(Xtr, Ytr, GDparams, lambda);
    Wf(:, :, n) = W;
    bf(:, n) = b;
    [alpha, wCur] = weightUpdate(Xtr, ytr, W, b, wCur);
    alphaf(n) = alpha;
    GDparams.eta = decay*GDparams.eta;
end

% derive final classification
Pf = zeros(size(Yte));
for n = 1 : n_trial
    [~, k_star] = ComputeAccuracy(Xte, yte, Wf(:, :, n), bf(:, n));
    Pf = Pf + k_star * alphaf(n);
end
acc_te = easyaccuracy(Pf, yte);

% print accuracy of the network
% acc_tr = ComputeAccuracy(Xtr, ytr, W, b);
% disp(['training accuracy:' num2str(acc_tr*100) '%'])
% acc_te = ComputeAccuracy(Xte, yte, W, b);
disp(['test accuracy:' num2str(acc_te*100) '%'])

% plot cost score
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

