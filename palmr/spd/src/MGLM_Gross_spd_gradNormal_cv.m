function [minloc_row, minloc_col] = MGLM_Gross_spd_gradNormal_cv(X, Y, Xval, Yval, lambdaSet, rhoSet, opts)
% MGLM_GROSS_SPD_GRADNORMAL_CV performs cross validation to choose the optimal
% parameters lambda and rho for PALMR model
%
%           X: a nbasis-by-tnsample matrix.
%           Y: a n-by-n-by-tnsample array storing SPD matrices.
%        Xval: a nbasis-by-nval matrix.
%        Yval: a n-by-n-by-nval array storing SPD matrices.
%   lambdaSet: a set of candidate parameters for lambda
%      rhoSet: a set of candidate parameters for rho
%        opts: other parameters
%
%  minloc_row: index of the selected parameter for lambda   
%  minloc_col: index of the selected parameter for rho
%
%
% Written by Xiaowei Zhang 
% 2015/05/13
% updated on 2017/02/14

if nargin < 4
    opts = [];
end

Numlambda = length(lambdaSet); 
Numrho = length(rhoSet);
Error = zeros(Numlambda, Numrho);

% start of cross-validation
for j=1:Numlambda
    for k=1:Numrho
        opts.lambda = lambdaSet(j);
        opts.rho = rhoSet(k);
        
        [phat, Vhat, ~, ~, ~, ~] = MGLM_Gross_spd_gradNormal(X, Y, opts);
        
        % error on validation data
        Y_hat = predSPD(phat, Vhat, Xval);
        Error(j,k) = MSGError_spd(Yval, Y_hat);
    end
end
[~,minloc] = min(Error(:));
[minloc_row, minloc_col] = ind2sub(size(Error), minloc);

end % end function