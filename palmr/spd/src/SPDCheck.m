function flag = SPDCheck(X, c)
% SPDCHECK checks if matrices in X are symmetric positive definite 
%           and whether the smallest eigenvalue is bigger than c.
%    
%   T = SPDCHECK(X)
%   T = SPDCHECK(X, c)
%
%      X: a n-by-n-by-N array.
%      c: a scalar. Default value is eps.
%   flag: equals to  0 or 1 (all matrices are symmetric positive definite)
%
% 
% Written by Xiaowei Zhang 
% 2015/05/12
% updated on 2017/02/14

if nargin < 2
    c = eps;
end

flag = 1;

% Check matrices are symmetric positive definite.
N = size(X,3); 
T = zeros(N,1);
for i=1:N
    temp = abs(X(:,:,i) - X(:,:,i)');
    if max(temp(:)) > 1e-5 || (min( eig(X(:,:,i)) ) <= c)
        T(i) =  1;
    end
end

if any(T)
    flag = 0;
end


