clear;
addpath /Users/MMGF2/Desktop/deep_learning_in_data_science/assign/assign1/Datasets/cifar-10-batches-mat/;

% read in training, validation and test data
[Xtr, Ytr, ytr] = LoadBatch('data_batch_1.mat');
[Xva, Yva, yva] = LoadBatch('data_batch_2.mat');
[Xte, Yte, yte] = LoadBatch('test_batch.mat');

% initialize W and b
mean = 0;
std = 0.01;
K = size(Ytr, 1);
d = size(Xtr, 1);
W = mean + randn(K, d)*std;
b = mean + randn(K, 1)*std;
lambda = 0;

% check if analytical gradient is correct
batch_size = 50;
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(Xtr(:, 1 : batch_size), ...
    Ytr(:, 1 : batch_size), W, b, lambda, 1e-6);
P = EvaluateClassifier(Xtr(:, 1 : batch_size), W, b);
[grad_W, grad_b] = ComputeGradients(Xtr(:, 1 : batch_size), ...
    Ytr(:, 1 : batch_size), P, W, lambda);
gradcheck_b = max(abs(ngrad_b - grad_b)./max(0, abs(ngrad_b) + abs(grad_b)))
gradcheck_W = max(max(abs(ngrad_W - grad_W)./max(0, abs(ngrad_W) + abs(grad_W))))

% perform the mini-batch gradient descent algorithm
GDparams = setGDparams(100, 0.1, 40);
Jtr = zeros(1, GDparams.n_epochs);
Jva = zeros(1, GDparams.n_epochs);
for i = 1 : GDparams.n_epochs
    Jtr(i) = ComputeCost(Xtr, Ytr, W, b, lambda);
    Jva(i) = ComputeCost(Xva, Yva, W, b, lambda);
    [W, b] = MiniBatchGD(Xtr, Ytr, GDparams, W, b, lambda);
end

% print accuracy of the network
acc_tr = ComputeAccuracy(Xtr, ytr, W, b);
disp(['training accuracy:' num2str(acc_tr*100) '%'])
acc_te = ComputeAccuracy(Xte, yte, W, b);
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

% visualize weight matrix as class template images
for i = 1 : K
im = reshape(W(i, :), 32, 32, 3);
s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure()
montage(s_im, 'size', [1, K])
