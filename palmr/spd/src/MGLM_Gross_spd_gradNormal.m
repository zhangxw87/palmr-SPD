function [p, V, Objval, Yhat, Yc, G] = MGLM_Gross_spd_gradNormal(X, Y, opts)
% MGLM_GROSS_SPD_GRADNORMAL performs MGLM on SPD manifold data with
% potential gross error. The algorithm uses the normaized gradient for line
% search
%
%   [p, V, Objval, Y_hat, G] = MGLM_GROSS_SPD_GRADNORMAL(X, Y)
%   [p, V, Objval, Y_hat, G] = MGLM_GROSS_SPD_GRADNORMAL(X, Y, OPTS)
%
%        X: a nbasis-by-tnsample matrix.
%        Y: a n-by-n-by-tnsample array storing SPD matrices.
%
%        p: a n-by-n SPD matrix which is a intercept point.
%        V: a n-by-n-by-nbasis array storing tangent vectors of p (each vector is a symmetric matrix).
%   Objval: a column vector storing the history of the objective function values.
%     Yhat: a n-by-n-by-tnsample array storing SPD matrices which are the predictions.
%       Yc: a n-by-n-by-tnsample array storing SPD matrices which are the corrected data.
%        G: a n-by-n-by-tnsample array storing symmetric matrices which are the gross error.
%
%
% 
% Written by Xiaowei Zhang 
% 2015/05/13
% updated on 2017/02/14

if nargin < 3
    opts = [];
end
% get or check parameters
opts = MGLM_Gross_spd_opts(opts);

[nbasis, tnsample] = size(X);
dim = size(Y,1);

if tnsample ~= size(Y,3)
    error('Different number of samples in X and Y.')
end

lambda = opts.lambda;
rho = opts.rho;
prox_para = opts.prox_para;
tol = opts.tol;
maxit = opts.maxit;

% Initial intercept
switch opts.init
    case 0
        p = karcher_mean_spd(Y);
    case 1
        p = Y(:,:,1);
    otherwise
        p = Y(:, :, randi(tnsample));  % a random training data as initial intercept
end

% Initial gross error and basis vectors
G = zeros([dim dim tnsample]);
V = zeros([dim dim nbasis]);
% V can be random symmetric matrices
% V = randn([dim dim nbasis]);
% V = Projection2TpM_spd(V);
% G = Projection2TpM_spd(G);

% initial objective function value
Objval = zeros(maxit+1,1);
Yhat = predSPD(p,V,X);
Yc = data_correct(Y, G);
Objval(1) = Objeval_spd(Y, Yhat, Yc, p, V, G, lambda, rho);

if opts.verbose    
    fprintf('Initial objective function value = %8.6f. \n', Objval(1));
    fprintf('Iteration \t Obj. fun. val. \t #line_search\n');
end

% start of main iteration
step = 1; 
MaxGradNorm = 1;
for niter = 1:maxit
    
    T_Yhat_Yc = LogMapSPD(Yhat, Yc);
    for i = 1:tnsample
        T_Yhat_Yc(:, :, i) = ParalTrans_spd(T_Yhat_Yc(:, :, i), Yhat(:, :, i), p);
    end
    
    % compute gradients w.r.t. p
    gradp = -sum(T_Yhat_Yc, 3);
    gradp_norm = Norm_TpM_spd(p, gradp);
    if gradp_norm > MaxGradNorm
        gradp = MaxGradNorm * (gradp ./ gradp_norm);
    end
    
    % compute gradients w.r.t. V
    gradV = zeros(size(V));
    for j = 1:nbasis
        gradV(:,:,j) = -weightedSum(T_Yhat_Yc, X(j,:));
        gradVj_norm = Norm_TpM_spd(p, gradV(:,:,j));
        if gradVj_norm > MaxGradNorm
            gradV(:,:,j) = MaxGradNorm * (gradV(:,:,j) ./ gradVj_norm);
        end
    end
        
    % start of inner iteration for optimal step size
    flag = 0;
    V_new = zeros(size(V)); G_new = zeros(size(G)); step = step / 2;
    for k = 1:50
        % update p
        p_new = ExpMapSPD(p, -gradp / (step*prox_para));
        
        % update V
        for j = 1:nbasis
            gradVj = V(:, :, j) - gradV(:, :, j) / (step*prox_para);
            gradVj_norm = Norm_TpM_spd(p, gradVj);
            if gradVj_norm <= (lambda / (step*prox_para))
                V_new(:, :, j) = zeros(dim);
            else
                V_new(:, :, j) = (1 - lambda / ((step*prox_para) * gradVj_norm) ) .* gradVj;
            end
        end
        
        % parallel transport V_new form tangent space of p to p_new
        for j = 1:nbasis
            V_new(:, :, j) = ParalTrans_spd(V_new(:, :, j), p, p_new);
        end
        Yhat_new = predSPD(p_new,V_new,X);  % update the prediction
        
        % compute gradients w.r.t G
        T_Yc_Yhat = LogMapSPD(Yc, Yhat_new);
        gradG = zeros(size(G)); 
        for i = 1:tnsample
            gradG(:, :, i) = -ParalTrans_spd(T_Yc_Yhat(:, :, i), Yc(:, :, i), Y(:, :, i));
            gradGi_norm = Norm_TpM_spd(Y(:, :, i), gradG(:, :, i));
            if gradGi_norm > MaxGradNorm
                gradG(:, :, i) = MaxGradNorm * (gradG(:, :, i) ./ gradGi_norm);
            end
        end
        
        for i = 1:tnsample
            gradGi = G(:, :, i) - gradG(:, :, i) / (step*prox_para);
            
            gradGi_norm = Norm_TpM_spd(Y(:, :, i), gradGi);
            if gradGi_norm <= (rho / (step*prox_para))
                G_new(:, :, i) = zeros(dim);
            else
                G_new(:, :, i) = (1 - rho / ((step*prox_para)*gradGi_norm)) .* gradGi;
            end
        end 
        Yc_new = data_correct(Y, G_new);
        
        % evaluate objective function
        Objval_new = Objeval_spd(Y, Yhat_new, Yc_new, p_new, V_new, G_new, lambda, rho);
        
        if Objval(niter) > Objval_new
            p = p_new;
            V = V_new;
            G = G_new;
            Yhat = Yhat_new;
            Yc = Yc_new;
            Objval(niter+1) = Objval_new;
            
            flag = 1;
            break
        end
        
        step = 2 * step;  % increase the proximal parameters
    end
    
    if ~flag
        fprintf('Can''t find proximal parameter to decrease the objective function value.\n');
        break;
    end
    
    if opts.verbose
        fprintf('    %i \t\t   %8.6f   \t   k = %i \n', niter, Objval(niter+1), k);
    end
    
    % stopping criterion
    if (abs(Objval(niter+1) - Objval(niter)) <= tol) && niter >= 50;
        break;
    end
end

if niter < maxit
    Objval(niter+2:end) = [];
% else
%    warning('The stopping criterion is not satisfied within %i iterations', maxit);
%    fprintf('The stopping criterion is not satisfied within %i iterations. \n', maxit);
end

if opts.verbose
    fprintf('Objective function value = %8.6f after %i iterations. \n', Objval(end-1), niter);
end

end

%% training data correction
function Yc = data_correct(Y, G)

Yc = zeros(size(Y));
for i=1:size(Y,3)
    Yc(:, :, i) = ExpMapSPD(Y(:, :, i), G(:, :, i));
end

end