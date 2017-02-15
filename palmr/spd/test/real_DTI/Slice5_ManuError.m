% Prediction of tensors on slice 5 using training data with 20% manual
% gross error
%
% Written by Xiaowei Zhang 
% updated on 2017/02/15

clear all; clc;
addpath('../../../../riem_mglm_0.12')

% load mask for slice 1
load('../../CMIND/data/wm_mask.mat')
mask = squeeze( mask(64, :, :) );
[mask_dim1, mask_dim2] = size(mask);

% path to tensors with complete pre-processing
tensor_file_format = '../../CMIND/data/voxel_tensors_mat_iter6/slice5/tensor_64_%d_%d.mat';

% load age and gender
load('../../CMIND/data/cmind_age_gender.mat');

% NumSample = 58; 
NumTrain = 40;
age = double(age);
indx = (age/365 >= 8);

% independent variables 
X = [age; gender];
X = double( X(:, indx) );
Norm_val = norm(X(1,:));
X(1,:) = X(1,:) / Norm_val; % normalize age feature

% male and female indices
male_ind = find(X(2,:) == 1);
female_ind = find(X(2,:) == -1);

rng('shuffle');
nrepeat = 10;  % number of repetitions

%%%%% gross error parameters %%%%%
  noise_g = 5;
  rate_g = 0.2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters for PALMR model
opts_wg.prox_para = 100;
opts_wg.lambda = 0.1;
opts_wg.rho = 1; 
opts_wg.maxit = 100;

% parameters for MGLM model
opts_ng.prox_para = 1;
opts_ng.lambda = 0;
opts_ng.rho = 1e+100;
opts_ng.maxit = 100;

% initialization of variables
phat_wg = zeros(mask_dim1,mask_dim2,3,3);
Vhat_wg = zeros(mask_dim1,mask_dim2,3,3,2);
Yhat_wg = zeros(mask_dim1,mask_dim2,3,3,NumTrain);
Yc_wg = zeros(mask_dim1,mask_dim2,3,3,NumTrain);
Ghat_wg = zeros(mask_dim1,mask_dim2,3,3,NumTrain);
Gtrue = zeros(mask_dim1,mask_dim2,3,3,NumTrain);

phat_ng = zeros(mask_dim1,mask_dim2,3,3);
Vhat_ng = zeros(mask_dim1,mask_dim2,3,3,2);
Yhat_ng = zeros(mask_dim1,mask_dim2,3,3,NumTrain);

% initialization of prediction errors
FA_pred_error_noise = zeros(mask_dim1,mask_dim2,nrepeat);
PALMR_pred_error_noise = zeros(mask_dim1,mask_dim2,nrepeat);
PALMR_FA_error_noise = zeros(mask_dim1,mask_dim2,nrepeat);
PALMR_G_error = zeros(mask_dim1,mask_dim2,nrepeat);
PALMR_G_Fscore = zeros(mask_dim1,mask_dim2,nrepeat);
PALMR_G_precision = zeros(mask_dim1,mask_dim2,nrepeat);
PALMR_G_recall = zeros(mask_dim1,mask_dim2,nrepeat);

MGLM_pred_error_noise = zeros(mask_dim1,mask_dim2,nrepeat);
MGLM_FA_error_noise = zeros(mask_dim1,mask_dim2,nrepeat);

% repeat experiments for 10 times
for r = 1:nrepeat
    % generate random splitting
    fprintf('We are processing the %i-th repeatition ...\n',r);
    [male_ind_tr, ~] = datasample(male_ind, 20, 'Replace', false);
    male_ind_te = setdiff(male_ind, male_ind_tr);
    
    [female_ind_tr, ~] = datasample(female_ind, 20, 'Replace', false);
    female_ind_te = setdiff(female_ind, female_ind_tr);
    ind_tr = sort([male_ind_tr female_ind_tr]);
    ind_te = sort([male_ind_te female_ind_te]);
    
    Xtr = X(:,ind_tr);
    Xte = X(:,ind_te);
    
    for i = 1:mask_dim1
        for j = 1:mask_dim2
            if mask(i, j) == 1 % only process voxels within white matter mask 
                fprintf('\n=============================================\n')
                fprintf('We are processing voxel i = %i, j = %i \n', i, j);
                file = sprintf(tensor_file_format, i, j);
                load(file)
                Y = double( tensor(indx, :, :) );
                Y = permute(Y, [2:3,1]); % reshape Y into size 3 x 3 x num_data
                Ytr_true = Y(:,:,ind_tr);
                Yte = Y(:,:,ind_te);
                
                % add gross error to Ytr
                ind_g = randperm(NumTrain); 
                Ytr = Ytr_true;
                for k = 1:ceil(NumTrain * rate_g)
                    [Ytr(:, :, ind_g(k)), Gtrue(i, j, :, :, ind_g(k))] = addGrossSPD(Ytr_true(:, :, ind_g(k)), noise_g);
                end
                
                % Multivariate geodesic regression with gross error
                [phat, Vhat, ~, Yhat, Yc, Ghat] = MGLM_Gross_spd(Xtr, Ytr, opts_wg);
                phat_wg(i,j,:,:) = phat;
                Vhat_wg(i,j,:,:,:) = Vhat;
                Yhat_wg(i,j,:,:,:) = Yhat;
                Yc_wg(i,j,:,:,:) = Yc;
                
                Ypred_wg = predSPD(phat, Vhat, Xte);
                PALMR_pred_error_noise(i,j,r) = MSGError_spd(Ypred_wg, Yte);
                
                % compute the precision and recall of identifying gross
                % errors
                ngerr = 0; TP = 0; FN = 0;
                for t = 1:NumTrain
                    if norm(Ghat(:,:,t),'fro') < 1e-8
                        Ghat(:,:,t) = zeros(3);
                    end
                    u = Gtrue(i,j,:,:,t); u = u(:);
                    v = Ghat(:,:,t); v = v(:);
                    
                    if any(v)
                        Ghat(:,:,t) = LogMapSPD(Yc(:,:,t), Ytr(:,:,t));
                        ngerr = ngerr + 1;
                    end
                    
                    if any(u) == 1 || any(v) == 1
                        temp = squeeze(Gtrue(i,j,:,:,t));
                        PALMR_G_error(i,j,r) = PALMR_G_error(i,j,r) + ...
                            MSGError_TpM_spd(Ytr_true(:,:,t), temp, Yc(:,:,t), Ghat(:,:,t));
                    end
                    
                    if any(u) == 1 && any(v) == 1
                        TP = TP + 1;
                    end
                    
                    if any(u) == 1 && any(v) == 0
                        FN = FN + 1;
                    end
                end
                Ghat_wg(i,j,:,:,:) = Ghat;
                
                PALMR_G_error(i,j,r) = PALMR_G_error(i,j,r) / NumTrain;
                PALMR_G_precision(i,j,r) = TP / ngerr;
                PALMR_G_recall(i,j,r) = TP / (TP + FN);
                PALMR_G_Fscore(i,j,r) = 2 * PALMR_G_precision(i,j,r) * PALMR_G_recall(i,j,r) / ...
                    ( PALMR_G_precision(i,j,r) + PALMR_G_recall(i,j,r) );
                
                % Multivariate geodesic regression without gross error
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % The original implementation of MGLM often results in
                % 'Numerical Error'. In such case, we can use palmr with 
                % specific parameters as an alternative choise, since MGLM 
                % is a specical case of PALMR when lambda = 0 and rho = inf.
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                try
                    [phat2, Vhat2, ~, Yhat2, ~] = mglm_spd(Xtr, Ytr);
                catch
                    % alternative choice
                    [phat2, Vhat2, ~, Yhat2, ~, ~] = MGLM_Gross_spd(Xtr, Ytr, opts_ng);
                end
                phat_ng(i,j,:,:) = phat2;
                Vhat_ng(i,j,:,:,:) = Vhat2;
                Yhat_ng(i,j,:,:,:) = Yhat2;
                
                Ypred_wg2 = predSPD(phat2, Vhat2, Xte);
                MGLM_pred_error_noise(i,j,r) = MSGError_spd(Ypred_wg2, Yte);
                
                % Fractional Anisotropy regression model
                FA_tr = Frac_Anisotropy(Ytr);
                FA_te = Frac_Anisotropy(Yte);
                [v, ~] = FA_MGLM_spd(Xtr, FA_tr);
                FA_pred = v * Xte;
                FA_pred_error_noise(i,j,r) = median( abs(FA_pred - FA_te) ./ FA_te );
                
                FA_palmr = Frac_Anisotropy(Ypred_wg);
                FA_mglm  = Frac_Anisotropy(Ypred_wg2);
                PALMR_FA_error_noise(i,j,r) = median( abs(FA_palmr - FA_te) ./ FA_te );
                MGLM_FA_error_noise(i,j,r) = median( abs(FA_mglm - FA_te) ./ FA_te );
            
                % print results
                fprintf('Prediction error:   Geodesic \t FA \n');
                fprintf('     PALMR model:   %.5f    %.5f \n', PALMR_pred_error_noise(i,j,r), ...
                    PALMR_FA_error_noise(i,j,r));
                fprintf('      MGLM model:   %.5f    %.5f \n', MGLM_pred_error_noise(i,j,r), ...
                    MGLM_FA_error_noise(i,j,r));
                fprintf('        FA model:     --       %.5f \n', FA_pred_error_noise(i,j,r));
                fprintf('Estimation of G\n\tError: %.5f\tPrecision: %.5f\tRecall: %.5f\n', ...
                    PALMR_G_error(i,j,r), PALMR_G_precision(i,j,r), PALMR_G_recall(i,j,r));
            end
        end
    end
    save ../../results/Slice5_ManuError_result.mat FA_pred_error_noise PALMR_FA_error_noise ...
        PALMR_pred_error_noise  MGLM_FA_error_noise MGLM_pred_error_noise;
end

exit;