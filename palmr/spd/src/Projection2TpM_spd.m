function V = Projection2TpM_spd(V)
% PROJ_TPM_SPD projects a set of tangent vectors V onto TpM, so that the output is a set of symmetric matrices.
%
%   V = PROJ_TPM_SPD(V)
%
%   V: a n-by-n-by-N array
%
%
% 
% Written by Xiaowei Zhang 
% 2015/05/13
% updated on 2017/02/14

for i = 1:size(V,3)
    V(:,:,i) = (V(:,:,i) + V(:,:,i)')/2;
end