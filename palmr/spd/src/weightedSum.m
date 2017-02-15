function S = weightedSum(X, w)
% WEIGHTEDSUM sums matrices with weight w.
%
%   S = WEIGHTEDSUM(X, w)
%   
%   X: a m-by-n-by-N array.
%   w: a vector of length N storing weights for matrices.
%   S: a m-by-n matrix which is the weighted sum of X.
%
% 
% Written by Xiaowei Zhang 
% 2015/05/12
% updated on 2017/02/14

[m, n, N] = size(X);
if length(w) ~= N
    error('The lenght of weight vector must be the same as the number of matrices.');
end
w = reshape(w,[1 1 N]);
w = repmat(w, [m n 1]);
S = sum(X.*w,3);