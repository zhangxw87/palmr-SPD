function Yhat = predSPD(p, V, X)
% PREDSPD predicts Yhat based on intercept p, basis V and covariate X.
%
%   Yhat = PREDSPD(p, V, X);
%
%      p: a n-by-n SPD maxtrix which is the intercept. 
%      V: a n-by-n-by-nbasis array storing a set of symmetric matrices which are tangent vectors of p.
%      X: a nbasis-by-tnsample matrix.
%   Yhat: a n-by-n-by-tnsample array storing a set of SPD matrices which are the predictions.
%
% 
% Written by Xiaowei Zhang 
% 2015/05/13
% updated on 2017/02/14
    
tnsample = size(X, 2);
Yhat = zeros([size(p) tnsample]);

for i = 1:tnsample
    v = weightedSum(V, X(:,i));
    v = (v + v') / 2;
    Yhat(:,:,i) = ExpMapSPD(p, v);
end
