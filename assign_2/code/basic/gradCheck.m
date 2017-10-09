function [gradcheck_W, gradcheck_b] = gradCheck(batch_size, X, Y, lambda, W, b, delta)

% numerical gradients
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(X(:, 1 : batch_size), ...
    Y(:, 1 : batch_size), W, b, lambda, delta);

% analytical gradients
h = intervalues(X(:, 1 : batch_size), W, b);
P = EvaluateClassifier(h, W, b);
[grad_W, grad_b] = ComputeGradients(X(:, 1 : batch_size), ...
    Y(:, 1 : batch_size), P, h, W, b, lambda);

% relative error rate
eps = 0.001;
gradcheck_b1 = sum(abs(ngrad_b{1} - grad_b{1})/max(eps, sum(abs(ngrad_b{1}) + abs(grad_b{1}))));
gradcheck_W1 = sum(sum(abs(ngrad_W{1} - grad_W{1})/max(eps, sum(sum(abs(ngrad_W{1}) + abs(grad_W{1}))))));
gradcheck_b2 = sum(abs(ngrad_b{2} - grad_b{2})/max(eps, sum(abs(ngrad_b{2}) + abs(grad_b{2}))));
gradcheck_W2 = sum(sum(abs(ngrad_W{2} - grad_W{2})/max(eps, sum(sum(abs(ngrad_W{2}) + abs(grad_W{2}))))));
gradcheck_W = [gradcheck_W1, gradcheck_W2];
gradcheck_b = [gradcheck_b1, gradcheck_b2];

end