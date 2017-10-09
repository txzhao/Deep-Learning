function [sbar, mu, v] = BN_forward(s, eps, mu_av, v_av)
if nargin < 4
    mu = mean(s, 2);
    v = mean((s - repmat(mu, 1, size(s, 2))).^2, 2);
else
    mu = mu_av;
    v = v_av;
end
sbar = diag((v + eps).^(-0.5))*(s - repmat(mu, 1, size(s, 2)));

end