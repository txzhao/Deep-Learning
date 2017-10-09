function h = intervalues(X, W, b)

W1 = W{1};
b1 = b{1};
b1 = repmat(b1, 1, size(X, 2));
h = W1*X + b1;
h = max(0, h);

end

