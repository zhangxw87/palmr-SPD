function exp_p_v = ExpMapSPD(p, v)
% EXPMAPSPD maps tangent vector v onto SPD manifold.
%
%   exp_p_v = EXPMAPSPD(p,v)
%
%         p: a n-by-n SPD matrix.
%         v: a n-by-n symmetric matrix, which is a tangent vector of p.
%   exp_p_v: a n-by-n SPD matrix.
%
% Written by Xiaowei Zhang 
% 2015/05/11
% updated on 2017/02/14

if norm(v) < eps
    exp_p_v = p;
    return
end

[U, D] = eig(p);
phalf = U * sqrt(D);
invp_half = inv(phalf);
v = invp_half * v * invp_half';
[U, D] = eig(v);
phalf_U = phalf * U;
exp_p_v = phalf_U * diag(exp(diag(D))) * phalf_U';
exp_p_v = (exp_p_v + exp_p_v') / 2;

if ~SPDCheck(exp_p_v)
    exp_p_v = Projection2SPD(exp_p_v);
end
