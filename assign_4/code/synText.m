function y = synText(RNN, h0, x0, n, K)
% initialize parameters
W = RNN.W;
U = RNN.U;
V = RNN.V;
b = RNN.b;
c = RNN.c;
h = h0;
x = x0;
y = zeros(1, n);

for t = 1 : n
    a = W*h + U*x + b;
    h = tanh(a);
    o = V*h + c;
    p = exp(o);
    p = p/sum(p);
    
    % randomly select a character based on probability
    cp = cumsum(p);
    a = rand;
    ixs = find(cp - a >0);
    ii = ixs(1);
    
    % generate the next input
    x = oneHot(ii, K);
    y(t) = ii;
end
    
end