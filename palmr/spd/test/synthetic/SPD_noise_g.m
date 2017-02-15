% experiments on SPD manifolds investingating the effect of the magnitude
% of gross error
%
% Written by Xiaowei Zhang 
% updated on 2017/02/14

clear; clc;
addpath('../../../../riem_mglm_0.12/spd');
addpath('../../../../riem_mglm_0.12/common');
addpath('../../src');

nrepeat = 10; % number of repetitions for each parameter
ntest = 100; % number of testing data
nval = 100;  % number of validation data

% para for sythetic data
noise_gSet = [0 1 5 10 15 20];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This parameter is important for the convergence of PALMR, if there is any
% error saying "Input to EIG must not contain NaN or Inf.", change it to a
% larger value.
  prox_para = [10:10:40 100 100]; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% regularization para
lambdaSet = 0.1;
rhoSet = 0.05:0.05:1;

for i = 1:nrepeat
    fprintf('The %i-th (over %i) repetition starts.\n', i, nrepeat);
    for j = 1:length(noise_gSet)        
        para.noise_g = noise_gSet(j);
        fprintf('\tWe are now processing the %i-th parameter...\n', j);
        
        RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));
        
        % generate training data
        [p, X, Y, Yrdn, Ytrue, V, G] = synth_spd_data(para);
        
        % generate validation data
        Xval = randn(size(X,1), nval);
        Yval = predSPD(p, V, Xval);
        
        % generate testing data
        Xte = randn(size(X,1), ntest);
        Yte = predSPD(p, V, Xte);
        
        % Multivariate geodesic regression with gross error
        t1 = tic;
        % use validation set to select the optimal model parameter
        opts.prox_para = prox_para(j); % change this value
        opts.maxit = 50;
        [row, col] = MGLM_Gross_spd_cv(X, Y, Xval, Yval, lambdaSet(1), rhoSet, opts);
        fprintf('\t\tOptimal parameters: lambda = %.2f, rho = %.2f. \n',...
            lambdaSet(1), rhoSet(col));
        
        opts.lambda = lambdaSet(1);
        opts.rho = rhoSet(col);        
        opts.maxit = 100;
        fprintf('\t\tStart of geodesic regression with gross error:\n');
        [phat_wg, Vhat_wg, ~, Yhat_wg, Yc_wg, Ghat_wg] = MGLM_Gross_spd(X, Y, opts);
        OutputWithG.Time(i,j) = toc(t1);
        Yte_wg = predSPD(phat_wg, Vhat_wg, Xte);
        
        OutputWithG.MSGE_Ytrain(i,j) = MSGError_spd(Ytrue, Yhat_wg);
        OutputWithG.MSGE_Ytest(i,j) = MSGError_spd(Yte, Yte_wg);
        OutputWithG.MSGE_Yc(i,j) = MSGError_spd(Yrdn, Yc_wg);
        OutputWithG.MSGE_p(i,j) = MSGError_spd(p, phat_wg);
        OutputWithG.MSGE_V(i,j) = MSGError_TpM_spd(p, V, phat_wg, Vhat_wg);
        
        Ghat_wg2 = zeros(size(G));
        OutputWithG.MSGE_G(i,j) = 0;
        OutputWithG.MSGE_Gratio(i,j) = 0;
        
        for k = 1:size(G,3)
            u = G(:,:,k); u = u(:);
            v = Ghat_wg(:,:,k); v = v(:);
            
            if any(v)
                Ghat_wg2(:,:,j) = LogMapSPD(Yc_wg(:,:,j), Y(:,:,j));
            end
            OutputWithG.MSGE_G(i,j) = OutputWithG.MSGE_G(i,j) + ...
                MSGError_TpM_spd(Yrdn(:,:,k), G(:,:,k), Yc_wg(:,:,k), Ghat_wg2(:,:,k));
            
            if any(u) == any(v)
                OutputWithG.MSGE_Gratio(i,j) = OutputWithG.MSGE_Gratio(i,j) + 1;
            end
        end
        OutputWithG.MSGE_G(i,j) = OutputWithG.MSGE_G(i,j) / size(G,3);
        OutputWithG.MSGE_Gratio(i,j) = OutputWithG.MSGE_Gratio(i,j) / size(G,3);
        
        % Multivariate geodesic regression without gross error
        fprintf('\t\tStart of geodesic regression w/o gross error:\n')
        t2 = tic;
%         try
%             [phat_ng, Vhat_ng, ~, Yhat_ng, ~] = mglm_spd(X,Y);
%             Yte_ng = predSPD(phat_ng, Vhat_ng, Xte);
%         catch
            % alternative choice
            opts_ng.lambda = 0;
            opts_ng.rho = 1e+10;
            opts_ng.prox_para = prox_para(j);
            opts_ng.maxit = 100;
            [phat_ng, Vhat_ng, ~, Yhat_ng, ~, ~] = MGLM_Gross_spd(X, Y, opts_ng);
            Yte_ng = predSPD(phat_ng, Vhat_ng, Xte);
%         end
        OutputNoG.Time(i,j) = toc(t2);
        
        OutputNoG.MSGE_Ytrain(i,j) = MSGError_spd(Ytrue, Yhat_ng);
        OutputNoG.MSGE_Ytest(i,j) = MSGError_spd(Yte, Yte_ng);
        OutputNoG.MSGE_p(i,j) = MSGError_spd(p, phat_ng);
        OutputNoG.MSGE_V(i,j) = MSGError_TpM_spd(p, V, phat_ng, Vhat_ng);
    end
    save ../../results/SPD_noise_g_result.mat OutputNoG OutputWithG;
end

%% compute median errors and standard deviations
OutputWithG.MSGE_Ytrain_median = median(OutputWithG.MSGE_Ytrain);
OutputWithG.MSGE_Ytest_median = median(OutputWithG.MSGE_Ytest);
OutputWithG.MSGE_Yc_median = median(OutputWithG.MSGE_Yc);
OutputWithG.MSGE_p_median = median(OutputWithG.MSGE_p);
OutputWithG.MSGE_V_median = median(OutputWithG.MSGE_V);
OutputWithG.MSGE_G_median = median(OutputWithG.MSGE_G);
OutputWithG.MSGE_Gratio_median = median(OutputWithG.MSGE_Gratio);

OutputNoG.MSGE_Ytrain_median = median(OutputNoG.MSGE_Ytrain);
OutputNoG.MSGE_Ytest_median = median(OutputNoG.MSGE_Ytest);
OutputNoG.MSGE_p_median = median(OutputNoG.MSGE_p);
OutputNoG.MSGE_V_median = median(OutputNoG.MSGE_V);

save ../../results/SPD_noise_g_result.mat OutputNoG OutputWithG;

%% print results
fprintf('      Median Squared Geodesic Errors: \t PALMR  \t  MGLM \n');
fprintf('              Error on training data:   %.5f  \t %.5f \n', ...
    [OutputWithG.MSGE_Ytrain_median; OutputNoG.MSGE_Ytrain_median]);
fprintf('               Error on testing data:   %.5f  \t %.5f \n', ...
    [OutputWithG.MSGE_Ytest_median; OutputNoG.MSGE_Ytest_median]);
fprintf('               Error of estimating p:   %.5f  \t %.5f \n', ...
    [OutputWithG.MSGE_p_median; OutputNoG.MSGE_p_median]);
fprintf('               Error of estimating V:   %.5f  \t %.5f \n', ...
    [OutputWithG.MSGE_V_median; OutputNoG.MSGE_V_median]);
fprintf('                 Error of correction:   %.5f  \t   --  \n', ...
    OutputWithG.MSGE_Yc_median);
fprintf('    Error of estimating G error size:   %.5f  \t   --  \n', ...
    OutputWithG.MSGE_G_median);
fprintf('Error of estimating G error location:   %.5f  \t   --  \n', ...
    OutputWithG.MSGE_Gratio_median);