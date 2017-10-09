function g = BN_backward(g, mu, v, S, eps)

g_vb = g*diag((v + eps).^(-1.5)).*(S - repmat(mu, 1, size(S, 2)))';
g_vb = 0.5*sum(g_vb);
g_mub = -sum(g*diag((v + eps).^(-0.5)));
g = g*diag((v + eps).^(-0.5)) + 2/size(S, 2)*repmat(g_vb,...
        size(S, 2), 1).*(S - repmat(mu, 1, size(S, 2)))' + ...
        repmat(g_mub, size(S, 2), 1)/size(S, 2);

end