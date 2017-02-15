function r = Norm_TpM_spd(p, v)
% NORM_TPM_SPD calculates the norm of v in T_{P}M on SPD manifolds.
%
%   r = NORM_TPM_SPD(p, v)
%
%   p: a SPD matrix
%   v: a tangent vactor of p (a symmetric matrix)
%   r: a nonnegative real scalar (norm of v)
%
%
% 
% Written by Xiaowei Zhang 
% 2015/05/12
% updated on 2017/02/14

[U, D] = eig(p);
phalf = U * sqrt(D);
invp_half = inv(phalf);
v = invp_half * v * invp_half';
r = norm(v, 'fro');
