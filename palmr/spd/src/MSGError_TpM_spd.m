function val = MSGError_TpM_spd(p, V, phat, Vhat)
% MSGERROR_TpM_SPD returns the mean squared geodesic error between the ground truth V and
% the estimation Vhat.
%
%   mse_val = MSGERROR_TpM_SPD(p, V, phat, Vhat)
%
%      p: a n-by-n SPD matrix.
%      V: a n-by-n-by-nbasis array storing a set of symmetric matrices (tangent vectors of p).
%   phat: a n-by-n SPD matrix (estimation of p).
%   Vhat: a n-by-n-by-nbasis array storing a set of symmetric matrices (tangent vectors of phat).
%
% 
% Written by Xiaowei Zhang 
% 2015/05/13
% updated on 2017/02/14

nbasis = size(V,3);

if size(Vhat,3) ~= nbasis
    error('The number of tangent vectors in V and Vhat must be the same.');
end

val = 0;

for i = 1:nbasis
    v = ParalTrans_spd(Vhat(:, :, i), phat, p);
    val = val + Norm_TpM_spd(p, V(:, :, i)-v)^2;
end

val = val / nbasis;

end