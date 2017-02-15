function d = GeoDist_spd(X,Y)
% GEODIST_SPD returns Riemannian distance between X and Y on SPD manifold.
%
%   d = GEODIST_SPD(X,Y)
%
%   X: a SPD matrix
%   Y: a SPD matrix
%   d: a nonnegative scalar denoting the Riemannian distance
%
% Written by Xiaowei Zhang 
% 2015/05/12
% updated on 2017/02/14

v = LogMapSPD(X,Y);
d = Norm_TpM_spd(X, v);

if ~isreal(d)
    error('Error: the geodesic distance between two points must be nonnegative real value.')
end