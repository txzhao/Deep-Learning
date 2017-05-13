function RNN = backward_Pass(RNN, X, Y, a, h, p, n, m, eta)

grads = ComputeGradients(RNN, X, Y, a, h, p, n, m);

for f = fieldnames(RNN)'
    RNN.(f{1}) = RNN.(f{1}) - eta*grads.(f{1});
end