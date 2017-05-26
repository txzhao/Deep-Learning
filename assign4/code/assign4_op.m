clear;

% read in data
if exist('tweet_data.mat') == 2
    load('tweet_data.mat');
else
    tweet_data = [];
    for i = 2009 : 2017
        tweet_fname = ['condensed_' num2str(i) '.json'];
        jsonData = loadjson(tweet_fname);
        for n = 1 : length(jsonData)
            tweet_piece = [jsonData{n}.text '^'];
            tweet_data = [tweet_data, tweet_piece];
        end 
    end
end
tweet_chars = unique(tweet_data);

% create empty map containers
char_to_ind = containers.Map('KeyType', 'char', 'ValueType', 'int32');
ind_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');

% fill in characters and integers as keys and values
keySet = num2cell(tweet_chars);
valueSet = 1 : length(keySet);
newMap1 = containers.Map(keySet, valueSet);
newMap2 = containers.Map(valueSet, keySet);
char_to_ind = [char_to_ind; newMap1];
ind_to_char = [ind_to_char; newMap2];

% initialize hyper-parameters
K = length(keySet);             % length of unique characters
m = 100;                        % dimensionality of hidden states
eta = 0.1;                      % learning rate
decay = 0.97;
seq_length = 20;                % length of input sequence
sig = 0.01;     
RNN.b = zeros(m, 1);            % bias vectors       
RNN.c = zeros(K, 1);
RNN.U = sqrt(2/(m + K))*randn(m, K);        % xavier initialization
RNN.W = sqrt(1/m)*randn(m, m);
RNN.V = sqrt(2/(m + K))*randn(K, m);
M.W = zeros(size(RNN.W));
M.U = zeros(size(RNN.U));
M.V = zeros(size(RNN.V));
M.b = zeros(size(RNN.b));
M.c = zeros(size(RNN.c));

% training process using AdaGrad
if exist('tweet_X.mat') == 2
    load('tweet_X.mat');
else
    X_ind = zeros(1, length(tweet_data));
    for i = 1 : length(tweet_data)
        X_ind(i) = char_to_ind(tweet_data(i));
    end
    tweet_X = oneHot(X_ind, K);
end

tweet_Y = tweet_X;     
iter = 1;
n_epochs = 2;
sl = 0;
SL = [];
hprev = [];
min_set.loss = 500;
min_set.RNN = [];
min_set.h = [];
min_set.iter = 1;
for i = 1 : n_epochs
    [RNN, sl, iter, M, min_set, eta] = MiniBatchGD_op(RNN, tweet_X, tweet_Y, ...
        seq_length, K, m, eta, iter, i, M, char_to_ind, ind_to_char, sl(end), min_set, decay);
    SL = [SL, sl];
end

plot(1 : length(SL), SL)
