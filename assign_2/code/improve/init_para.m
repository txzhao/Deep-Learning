function [W, b] = init_para(m, d, K, std)

W1 = std*randn(m, d);
W2 = std*randn(K, m);
b1 = zeros(m, 1);
b2 = zeros(K, 1);
W = {W1, W2};
b = {b1, b2};

end