function [v, FA_yhat] = FA_MGLM_spd(X, FA_y, lambda)
% FA_MGLM_SPD performs MGLM on SPD manifold data with FA_y being the Fractional Anisotropy.
%
%   [v, FA_yhat] = MGLM_GROSS_SPD(X, FA_y)
%   [v, FA_yhat] = MGLM_GROSS_SPD(X, FA_y, lambda)
%
%        X: a nbasis-by-tnsample matrix.
%     FA_y: a 1-by-tnsample vector storing fractional anisotropy of SPD matrices.
%        v: a 1-by-nbasis coefficient vector.
%  FA_yhat: a 1-by-tnsample vector storing predicted fractional anisotropy.
%
%
% Written by Xiaowei Zhang 
% 2015/10/02
% updated on 2017/02/15

if nargin < 3
    lambda = 0.01;
end

[nbasis, tnsample] = size(X);

% centering data
FA_y = FA_y - mean(FA_y) * ones(1, tnsample);
X = X - mean(X,2) * ones(1, tnsample);

v = FA_y * X';
v = v / (X * X' + lambda * eye(nbasis));

FA_yhat = v * X; 

end