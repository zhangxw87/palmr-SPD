function [v, g, FA_yhat, FA_yc] = FA_MGLM_Gross_spd(X, FA_y, opts)
% FA_MGLM_GROSS_SPD performs MGLM with gross error on SPD manifold data 
% with FA_y being the Fractional Anisotropy.
%
%   [v, FA_yhat] = MGLM_GROSS_SPD(X, FA_y)
%   [v, FA_yhat] = MGLM_GROSS_SPD(X, FA_y, opts)
%
%        X: a nbasis-by-tnsample matrix.
%     FA_y: a 1-by-tnsample vector storing fractional anisotropy of SPD matrices.
%        v: a 1-by-tnsample coefficient vector.
%        g: a 1-by-tnsample vector storing recoved gross error.
%  FA_yhat: a 1-by-tnsample vector storing predicted fractional anisotropy.
%
%
% Written by Xiaowei Zhang 
% 2015/10/02
% updated on 2017/02/15

if nargin < 3
    opts = [];
end
opts = FA_MGLM_Gross_spd_opts(opts);
lambda = opts.lambda;
rho = opts.rho;
tol = opts.tol;
maxit = opts.maxit;

[nbasis, tnsample] = size(X);
% centering data
FA_y = FA_y - mean(FA_y) * ones(1, tnsample);
X = X - mean(X,2) * ones(1, tnsample);

% Lipschitz constant
L = norm(X)^2 + lambda + 1;

% Initial points
g = zeros(1,tnsample);
gtilde = g;
v = zeros(1, nbasis);
vtilde = v;
t1 = 1;
Objval = 0.5 * (FA_y * FA_y');

for i = 1:maxit
    temp1 = gtilde + vtilde * X - FA_y;
    temp2 = gtilde - temp1 / L; 
    g_new = max(abs(temp2) - rho / L, 0) .* sign(temp2);
    v_new = vtilde - (temp1 * X' + lambda * vtilde) / L;
    
    Objval_new = 0.5 * norm(FA_y - v_new * X - g_new)^2 + 0.5 * lambda * (v_new * v_new')...
        + rho * sum(abs(g_new));
    if abs(Objval_new - Objval) < tol
        break;
    end
    
    t2 = (1 + sqrt(1 + 4 * t1^2)) / 2;
    gtilde = gtilde + (t1 - 1) * (g_new - g) / t2;
    vtilde = vtilde + (t1 - 1) * (v_new - v) / t2;
    
    t1 = t2;
    g = g_new;
    v = v_new;
end

g = g_new;
v = v_new;
FA_yhat = v * X;
FA_yc = FA_y - g;
end

function opts = FA_MGLM_Gross_spd_opts(opts)

if isfield(opts,'lambda')
    if opts.lambda < 0
        error('opts.lambda must be nonnegative.');
    end
else
    opts.lambda = 0.01;
end

if isfield(opts,'rho')
    if opts.rho < 0
        error('opts.rho1 must be nonnegative.');
    end
else
    opts.rho = 0.1;
end

if isfield(opts,'tol')
    if (opts.tol <= 0) || (opts.tol >= 1)
        error('opts.tol is tolerance on objective function value decrement. Should be in (0,1).');
    end
else
    opts.tol = 1e-5;
end

if isfield(opts,'maxit')
    if (opts.maxit <= 0) || (opts.maxit ~= floor(opts.maxit)) 
        error('opts.maxit should be a positive integer.');
    end
else
    opts.maxit = 500;
end

end

