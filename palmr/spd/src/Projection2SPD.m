function p = Projection2SPD(X, threshold)
% PROJECTION2SPD projects a matrix onto SPD manifolds.
%
%   p = PROJECTION2SPD(X)
%
%   X: a n-by-n matrix.
%   p: a n-by-n SPD matrix.
%
% 
% Written by Xiaowei Zhang 
% 2015/05/14
% updated on 2017/02/14

if nargin < 2
    threshold = eps;
end

% Make a matrix symmetric positive definite.
X = (X + X')/2;
n = size(X, 1);

[V, D] = eig(X);
D = diag(D);
ind = (D > threshold);
p = V(:,ind) * diag(D(ind)) * V(:,ind)'; 
p = (p + p') / 2;  % p is positive semi-definite

% Make spd matrix
if sum(ind) < length(D)
    p = p + threshold*eye(n);
end

end