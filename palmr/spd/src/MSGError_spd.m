function val = MSGError_spd(Y, Yhat)
% MSGERROR_SPD returns the mean squared geodisc error between the ground truth Y and
% the estimation Yhat.
%
%   val = MSGERROR_SPD(X,Y)
%
%      Y: a n-by-n-by-N array storing a set of SPD matrices.
%   Yhat: a n-by-n-by-N array storing a set of SPD matrices.
%    val: a nonnegative real value.
%
% 
% Written by Xiaowei Zhang 
% 2015/05/13
% updated on 2017/02/14

N = size(Y,3);

if size(Yhat,3) ~= N
    error('The number of data points in Y and Yhat must be the same.');
end

val = 0;

for i = 1:N
    val = val + GeoDist_spd(Y(:, :, i), Yhat(:, :, i))^2;
end

val = val / N;

end