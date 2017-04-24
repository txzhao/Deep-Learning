function acc = easyaccuracy(P, y)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - P:            K*N
%           - y:            1*N
% OUTPUT    - acc:          1*1

[~, k_star] = max(P);
acc = length(find(y - k_star == 0))/length(y);

end