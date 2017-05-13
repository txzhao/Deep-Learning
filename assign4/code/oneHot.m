function out = oneHot(label, K)
% DENOTE N as the number of images, K as the number of label kinds
% INPUT     - label:        1*N
% OUTPUT    - out:          K*N

N = length(label);
out = zeros(K, N);
for i = 1 : N
    out(label(i), i) = 1;
end

end