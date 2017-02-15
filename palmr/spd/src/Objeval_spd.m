function Objval = Objeval_spd(Y, Yhat, Yc, p, V, G, lambda, rho)
% OBJEVAL_SPD evaluates the objective function value of MGLM_GROSS on SPD. 
%
%   OBJVAL = OBJEVAL_SPD(Y, p, Yhat, Yc, V, G, lambda, rho)
%
%      Y: a n-by-n-by-tnsample array storing SPD matrices.
%   Yhat: a n-by-n-by-tnsample array storing predicted SPD matrices.
%     Yc: a n-by-n-by-tnsample array storing corrected SPD matrices.
%      p: a n-by-n SPD matrix which is the intercept.
%      V: a n-by-n-by-nbasis array storing a set of symmetric matrices 
%         which are tangent vectors of p.
%      G: a n-by-n-by-tnsample array storing symmetric matrices, each of 
%         which is a tangent vector of the corresponding Y.    
%   lambda, rho are regularization parameters.   
%
% 
% Written by Xiaowei Zhang 
% 2015/05/13
% updated on 2017/02/14

Objval = 0;
 
for i = 1:size(Y,3)
    Objval = Objval + 0.5 * GeoDist_spd(Yc(:,:,i), Yhat(:,:,i))^2 ...
        + rho * Norm_TpM_spd(Y(:, :, i), G(:, :, i));
end

for j = 1:size(V,3)
    Objval = Objval + lambda * Norm_TpM_spd(p, V(:, :, j));
end

if ~isreal(Objval) || (Objval < 0)
    error('Error: the objective function value must be nonnegative real value.')
end