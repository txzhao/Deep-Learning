function out = oneHot(label)
% DENOTE N as the number of images, K as the number of label kinds
% INPUT     - label:        1*N
% OUTPUT    - out:          K*N

K = length(unique(label));
N = length(label);
out = zeros(K, N);
for i = 1 : N
    out(label(i), i) = 1;
end

end