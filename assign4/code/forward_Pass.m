function [loss, a, h, o, p] = forward_Pass(RNN, X, Y, h0, n, K, m)
% initialize parameters
W = RNN.W;
U = RNN.U;
V = RNN.V;
b = RNN.b;
c = RNN.c;
ht = h0;
o = zeros(K, n);
p = zeros(K, n);
h = zeros(m, n);
a = zeros(m, n);
loss = 0;

for t = 1 : n
    at = W*ht + U*X(:, t) + b;
    a(:, t) = at;
    ht = tanh(at);
    h(:, t) = ht;
    o(:, t) = V*ht + c;
    pt = exp(o(:, t));
    p(:, t) = pt/sum(pt);

    loss = loss - log(Y(:, t)'*p(:, t));
end

h = [h0, h];

end