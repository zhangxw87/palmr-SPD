function V = LogMapSPD(X,Y)
% LOGMAPSPD returns logmap(X,Y) for SPD manifolds.
%
%   V = LOGMAPSPD(X,Y)
%
%   X: a n-by-n-by-N array storing a set of SPD matrices.
%   Y: a n-by-n-by-N array storing a set of SPD matrices.
%   V: a n-by-n-by-N array where the V(:,:,i) stores the tangent vector 
%      from X(:,:,i) to Y(:,:,i).
%
% Written by Xiaowei Zhang 
% 2015/05/13
% updated on 2017/02/14

V = zeros(size(Y));
N = size(X,3);

if N > 1
    if size(Y,3) ~= N
        error('The number of SPD matrix in X and Y must be the same.')
    end
    
    for i = 1:N
        V(:,:,i) = logmap_pt2array_spd(X(:, :, i), Y(:, :, i));
    end
else
    V = logmap_pt2array_spd(X,Y);
end

end

function V = logmap_pt2array_spd(p,Y)
% LOGMAP_PT2ARRAY_SPD returns logmap(p,Y) for SPD manifolds.

[U, D] = eig(p);
phalf = U * sqrt(D);
% invp_half = inv(phalf);
invp_half = diag(1./sqrt(diag(D))) * U';

V = zeros(size(Y)); N = size(Y,3);

for i = 1:N
    if norm(p - Y(:,:,i)) < eps
        V(:,:,i) = zeros(size(p));
        continue
    end
    temp = invp_half * Y(:,:,i) * invp_half';
    temp = (temp + temp') / 2;
    [U, D] = eig(temp);
    phalf_U = phalf * U;
    V(:,:,i) = phalf_U * diag(log(max(diag(D), eps))) * phalf_U';
    V(:,:,i) = ( V(:,:,i) + V(:,:,i)' ) / 2;
    V(:,:,i) = real( V(:,:,i) );
end

end