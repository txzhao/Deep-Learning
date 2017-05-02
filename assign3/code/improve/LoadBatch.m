function [X, Y, y] = LoadBatch(filename)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - filename:     string
% OUTPUT    - X:            d*N
%           - Y:            K*N
%           - y:            1*N

indata = load(filename);
X = double(indata.data')/255;   % convert image from 0-256 to 0-1
y = double(indata.labels') + 1;
Y = oneHot(y);                  % convert labels to one-hot representation

end
