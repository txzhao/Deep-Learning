clear;
addpath H:\assign1\Datasets\cifar-10-batches-mat\;

% read in training, validation and test data
[Xtr, Ytr, ytr] = LoadBatch('data_batch_1.mat');
[Xva, Yva, yva] = LoadBatch('data_batch_2.mat');
[Xte, Yte, yte] = LoadBatch('test_batch.mat');
Xva = Xva(:, 1:1000);
Yva = Yva(:, 1:1000);
yva = yva(:, 1:1000);

% bias trick
Xtr = [Xtr; ones(1, size(Xtr, 2))];
Xva = [Xva; ones(1, size(Xva, 2))];
Xte = [Xte; ones(1, size(Xte, 2))];

% initialization
mean = 0;
std = 0.01;
K = size(Ytr, 1);
d = size(Xtr, 1);
W = mean + randn(K, d)*std;
lambda = 0;
delta = 1;
decay = 0.9;

% check if analytical gradient is correct
% batch_size = 50;
% [ngrad_b, ngrad_W] = ComputeGradsNumSlow(Xtr(:, 1 : batch_size), ...
%     Ytr(:, 1 : batch_size), W, b, lambda, 1e-6);
% P = EvaluateClassifier(Xtr(:, 1 : batch_size), W, b);
% [grad_W, grad_b] = ComputeGradients(Xtr(:, 1 : batch_size), ...
%     Ytr(:, 1 : batch_size), P, W, lambda);
% gradcheck_b = max(abs(ngrad_b - grad_b)./max(0, abs(ngrad_b) + abs(grad_b)))
% gradcheck_W = max(max(abs(ngrad_W - grad_W)./max(0, abs(ngrad_W) + abs(grad_W))))

% perform the mini-batch gradient descent algorithm
GDparams = setGDparams(100, 0.007, 60);
Jtr = zeros(1, GDparams.n_epochs);
Jva = zeros(1, GDparams.n_epochs);
for i = 1 : GDparams.n_epochs
    Jtr(i) = ComputeCost(Xtr, Ytr, W, lambda, delta);
    Jva(i) = ComputeCost(Xva, Yva, W, lambda, delta);
    
    % early stopping
%     if i ~= 1 && Jva(i - 1) - Jva(i) < 0
%         break;
%     end

    W = MiniBatchGD(Xtr, Ytr, GDparams, W, delta, lambda);
    GDparams.eta = decay*GDparams.eta;
end

% print accuracy of the network
acc_tr = ComputeAccuracy(Xtr, ytr, W);
disp(['training accuracy:' num2str(acc_tr*100) '%'])
acc_te = ComputeAccuracy(Xte, yte, W);
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

