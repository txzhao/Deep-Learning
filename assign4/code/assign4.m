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
eta = 0.01;                      % learning rate
seq_length = 25;                % length of input sequence
sig = 0.01;     
RNN.b = zeros(m, 1);            % bias vectors       
RNN.c = zeros(K, 1);
RNN.U = randn(m, K)*sig;        % weight matrices
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;
n_epochs = 300000;

% training process using AdaGrad
for e = 1 : n_epochs
    X_chars = book_data(e : e + seq_length - 1);
    Y_chars = book_data(e + 1 : e + seq_length);
    X_ind = zeros(1, seq_length);
    Y_ind = zeros(1, seq_length);
    for i = 1 : seq_length
        X_ind(i) = char_to_ind(X_chars(i));
        Y_ind(i) = char_to_ind(Y_chars(i));
    end
    X = oneHot(X_ind, K);
    Y = oneHot(Y_ind, K);
    if e == 1
        hprev = zeros(m, 1);
    else
        hprev = h(:, end);
    end
    [loss, a, h, o, p] = forward_Pass(RNN, X, Y, hprev, seq_length, K, m);
    RNN = backward_Pass(RNN, X, Y, a, h, p, seq_length, m, eta);
    
    if e == 1
        smooth_loss = loss;
    end
    smooth_loss = 0.999*smooth_loss + 0.001*loss;
    
    if mod(e, 100) == 0
        disp(['iter = ' num2str(e) ', smooth loss = ' num2str(smooth_loss)]);
    end
end

% % gradients check
% batch_size = 5;
% dh = 1e-4;
% gradCheck(batch_size, X, Y, RNN, dh, m, K);





% y = synText(RNN, h0, X(:, 1), seq_length, K);

% x0 = zeros(80, 1);
% x0(1) = 1;
% h0 = zeros(100, 1);
% h0([1 5 6 7]) = 1;
% n = 10;
% y = synText(RNN, h0, x0, n, K);
% c = [];
% for i = 1 : seq_length
%     c = [c ind_to_char(y(i))];
% end