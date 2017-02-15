function FA_val = Frac_Anisotropy(Y)
% FRAC_ANISOTROPY calculates the fractional anisotropy of DTI data in Y
%
%        Y: a 3-by-3-by-m array storing m SPD data 
%   FA_val: a 1-by-m vector storing FA value for each SPD in Y 
%
%
% Written by Xiaowei Zhang 
% 2015/10/02

m = size(Y,3);
FA_val = zeros(1,m);

for i = 1:m
    d = eig(Y(:,:,i));
    FA_val(i) = 3 * std(d,1) / (sqrt(2) * norm(Y(:,:,i),'fro'));
end

end