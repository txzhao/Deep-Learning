function [RNN, M] = backward_Pass(RNN, X, Y, a, h, p, n, m, eta, M)

grads = ComputeGradients(RNN, X, Y, a, h, p, n, m);
eps = 1e-8;

for f = fieldnames(RNN)'
%     % clip gradients to avoid exploding gradient
%     grads.(f{1}) = max(min(grads.(f{1}), 5), -5);

    M.(f{1}) = M.(f{1}) + grads.(f{1}).^2;
    RNN.(f{1}) = RNN.(f{1}) - eta*(grads.(f{1})./(M.(f{1}) + eps).^(0.5));
end

end