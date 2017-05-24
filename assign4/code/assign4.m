clear;

% read in data
book_fname = 'goblet_book.txt';
fid = fopen(book_fname, 'r');
book_data = fscanf(fid, '%c');
fclose(fid);
book_chars = unique(book_data);

% create empty map containers
char_to_ind = containers.Map('KeyType', 'char', 'ValueType', 'int32');
ind_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');

% fill in characters and integers as keys and values
keySet = num2cell(book_chars);
valueSet = 1 : length(keySet);
newMap1 = containers.Map(keySet, valueSet);
newMap2 = containers.Map(valueSet, keySet);
char_to_ind = [char_to_ind; newMap1];
ind_to_char = [ind_to_char; newMap2];

% initialize hyper-parameters
K = length(keySet);             % length of unique characters
m = 100;                        % dimensionality of hidden states
eta = 0.1;                      % learning rate
seq_length = 25;                % length of input sequence
sig = 0.01;     
RNN.b = zeros(m, 1);            % bias vectors       
RNN.c = zeros(K, 1);
RNN.U = randn(m, K)*sig;        % weight matrices
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;
M.W = zeros(size(RNN.W));
M.U = zeros(size(RNN.U));
M.V = zeros(size(RNN.V));
M.b = zeros(size(RNN.b));
M.c = zeros(size(RNN.c));

% gradients check
% X_ind = zeros(1, length(book_data));
% for i = 1 : length(book_data)        
%     X_ind(i) = char_to_ind(book_data(i));
% end
% X = oneHot(X_ind, K);
% batch_size = 15;
% dh = 1e-4;
% gradCheck(batch_size, X(:, 1 : seq_length), X(:, 2 : seq_length + 1), RNN, dh, m, K);

% training process using AdaGrad
if exist('X.mat') == 2
    load('X.mat');
else
    X_ind = zeros(1, length(book_data));
    for i = 1 : length(book_data)
        X_ind(i) = char_to_ind(book_data(i));
    end
    X = oneHot(X_ind, K);
end

Y = X;     
iter = 1;
n_epochs = 7;
SL = [];
for i = 1 : n_epochs
    [RNN, sl, iter, M] = MiniBatchGD(RNN, X, Y, seq_length, K, m, eta, iter, M, ind_to_char);
    SL = [SL, sl];
end

% plot the smooth loss
plot(1:length(SL), SL)

