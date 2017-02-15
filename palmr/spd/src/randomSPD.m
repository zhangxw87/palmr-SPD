function p = randomSPD(n, var)
% RANDOMSPD generates n-by-n random symmatrix positive matrix p.
%
%   p = RANDOMSPD(n)
%   p = RANDOMSPD(n,var)
%
%       n: the dimension of SPD matrix.
%     var: a parameter for variance. Larger var leads to larger variance.
%
% 
% Written by Xiaowei Zhang 
% 2015/05/12
% updated on 2017/02/14

if nargin < 2
    var = 3;
end

p = var * (rand(n) - 0.5);
p = p * p' + eye(n);


end
