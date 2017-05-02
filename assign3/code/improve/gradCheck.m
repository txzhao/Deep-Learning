function gradCheck(batch_size, X, Y, lambda, W, b, delta, k)

% numerical gradients
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(X(:, 1 : batch_size), ...
    Y(:, 1 : batch_size), W, b, lambda, delta);

% analytical gradients
[h, S, mu, v] = intervalues(X(:, 1 : batch_size), W, b, k);
P = EvaluateClassifier(h, W, b);
[grad_W, grad_b] = ComputeGradients(X(:, 1 : batch_size), ...
    Y(:, 1 : batch_size), P, h, S, W, lambda, k, mu, v);

% relative error rate
eps = 0.001;
for i = 1 : length(W)
    gradcheck_bi = sum(abs(ngrad_b{i} - grad_b{i})/max(eps, sum(abs(ngrad_b{i}) + abs(grad_b{i}))));
    gradcheck_bm = max(abs(ngrad_b{i} - grad_b{i})/max(eps, sum(abs(ngrad_b{i}) + abs(grad_b{i}))));
    gradcheck_Wi = sum(sum(abs(ngrad_W{i} - grad_W{i})/max(eps, sum(sum(abs(ngrad_W{i}) + abs(grad_W{i}))))));
    gradcheck_Wm = max(max(abs(ngrad_W{i} - grad_W{i})/max(eps, sum(sum(abs(ngrad_W{i}) + abs(grad_W{i}))))));
    disp(['error of grad_W' num2str(i) ': ' num2str(gradcheck_Wi)])
    disp(['max error of grad_W' num2str(i) ': ' num2str(gradcheck_Wm)])
    disp(['error of grad_b' num2str(i) ': ' num2str(gradcheck_bi)])
    disp(['max error of grad_b' num2str(i) ': ' num2str(gradcheck_bm)])
end

end