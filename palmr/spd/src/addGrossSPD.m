function [q, v] = addGrossSPD(p, noise)
% ADDGROSS_SPD adds gross noise to SPD matrix p by moving along a random geodesic curve 
%
%   [q, v] = ADDNOISE_SPD(p, noise)
%
%       p: a n-by-n SPD matrix.
%   noise: a positive scalar indicating the level of random noise.
%       q: a n-by-n SPD matrix obtained by adding noise to p.
%       v: a n-by-n symmetric matrix which is the tangent vector of p to q.
%
% 
% Written by Xiaowei Zhang 
% 2015/05/14
% updated on 2017/02/14

while true
    v = randn( size(p) );
    v = (v + v') / 2;  % random symmetric matrix
    
    normv = Norm_TpM_spd(p, v);
    if normv > eps
        v = (v / normv) * noise;
        break
    end
end

q = ExpMapSPD(p, v);