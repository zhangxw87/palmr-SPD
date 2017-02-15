function u = ParalTrans_spd(v, p, q)
% PARALTRANS_SPD transports a tangent vector v from TpM to TqM.
%
%   u = PARALTRANS_SPD(v, p, q)
%
%   v: a n-by-n symmetric matrix which is a tangent vector of p.
%   p: a n-by-n SPD. 
%   q: a n-by-n SPD. 
%   u: a n-by-n symmetric matrix which is a tangent vector of q.
%
% 
% Written by Xiaowei Zhang 
% 2015/05/13
% updated on 2017/02/14

w = LogMapSPD(p, q);
if norm(p - q) < 1e-20
    u = v;
else
    [U, D] = eig(p);
    phalf = U * sqrt(D);
    invp_half = inv(phalf);    
    y = invp_half * w * invp_half'/2;
    [U, D] = eig(y);
    r = U * diag(exp(diag(D))) * U';
    temp = phalf * r * invp_half;
    u = temp * v * temp';
    u = real( (u + u') / 2 );
end
