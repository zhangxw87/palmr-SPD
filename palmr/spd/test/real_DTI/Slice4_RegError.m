% Prediction of tensors on slice 4 using training data with 20%
% registration gross error
%
% Written by Xiaowei Zhang 
% updated on 2017/02/15

clear all; clc;
addpath('../../../../riem_mglm_0.12')

% load mask for slice 1
load('../../CMIND/data/wm_mask.mat')
mask = squeeze( mask(:, :, 24) );
[mask_dim1, mask_dim2] = size(mask);
NumVoxel = sum(mask(:));

%% pathes to tensors on slice 1

% path to tensors with registration error
tensor_file_format1 = '../../CMIND/data/voxel_tensors_mat_iter0/slice4/tensor_%d_%d_24.mat';
% path to tensors with complete pre-processing
tensor_file_format2 = '../../CMIND/data/voxel_tensors_mat_iter6/slice4/tensor_%d_%d_24.mat';

% load age and gender
load('../../CMIND/data/cmind_age_gender.mat');

% select samples with age >= 8
age = double(age);
indx = (age/365 >= 8);
NumTrain = 40;
NumSample = sum(indx);

% independent variables 
X = [age; gender];
X = double( X(:, indx) );
Norm_val = norm(X(1,:));      % normalize age feature
X(1,:) = X(1,:) / Norm_val;

% male and female indices
male_ind = find(X(2,:) == 1);
female_ind = find(X(2,:) == -1);

%% Tensor difference
% compute tensor difference
Tensor_diff = zeros(mask_dim1, mask_dim2, NumSample);
for i = 1:mask_dim1
    for j = 1:mask_dim2
        if mask(i, j) == 1
            file = sprintf(tensor_file_format2, i, j);
            load(file)
            Ytrue = double( tensor(indx, :, :) );
            Ytrue = permute(Ytrue, [2:3,1]);

            file = sprintf(tensor_file_format1, i, j);
            load(file)
            Y = double( tensor(indx, :, :) );
            Y = permute(Y, [2:3,1]);

            for k = 1:NumSample
                Tensor_diff(i,j,k) = GeoDist_spd(Ytrue(:,:,k), Y(:,:,k));
            end
        end
    end
end
Tensor_diff_med = median(Tensor_diff, 3);
Tensor_diff_min = min(Tensor_diff, [], 3);
Tensor_diff_max = max(Tensor_diff, [], 3);

% plot tensor  difference
figure(1)
subplot(1,3,1)
colormap('jet')
imagesc(rot90(Tensor_diff_max, 1))
colorbar
axis tight
title('Maximum deviation')

subplot(1,3,2)
colormap('jet')
imagesc(rot90(Tensor_diff_min, 1))
colorbar
axis tight
title('Minimum deviation')

subplot(1,3,3)
colormap('jet')
imagesc(rot90(Tensor_diff_med, 1))
colorbar
axis tight
title('Median deviation')

%% Prediction on voxels with minimum deviation >= threshold

% find voxels with large error
threshold = 0.7;
[rows, cols, ~] = find(Tensor_diff_min > threshold);
numROI = length(rows);

rng('shuffle');
nrepeat = 10; % number of train/testing splits
rate_g = 0.2; % percent of training samples containing registration error

% parameters for PALMR model
opts_wg.prox_para = 10;
opts_wg.lambda = 0.1;
opts_wg.rho = 0.5;
opts_wg.maxit = 100;
% opts_wg.verbose = 1;

% parameters for MGLM model
opts_ng.prox_para = 10;
opts_ng.lambda = 0;
opts_ng.rho = 1e+100;
opts_ng.maxit = 100;

% initialization of variables
phat_wg = zeros(numROI,3,3);
Vhat_wg = zeros(numROI,3,3,2);
Yhat_wg = zeros(numROI,3,3,NumTrain);
Yc_wg = zeros(numROI,3,3,NumTrain);

phat_ng = zeros(numROI,3,3);
Vhat_ng = zeros(numROI,3,3,2);
Yhat_ng = zeros(numROI,3,3,NumTrain);

% initialization of prediction errors
FA_pred_error_roi = zeros(numROI,nrepeat);
PALMR_pred_error_roi = zeros(numROI,nrepeat);
PALMR_FA_error_roi = zeros(numROI,nrepeat);
PALMR_G_precision = zeros(numROI,nrepeat);
PALMR_G_recall = zeros(numROI,nrepeat);
MGLM_pred_error_roi = zeros(numROI,nrepeat);
MGLM_FA_error_roi = zeros(numROI,nrepeat);

% repeat experiments for 10 times
for r = 1:nrepeat
    % generate random splitting
    fprintf('We are processing the %i-th repeatition ...\n',r);
    [male_ind_tr, ~] = datasample(male_ind, NumTrain/2, 'Replace', false);
    male_ind_te = setdiff(male_ind, male_ind_tr);
    
    [female_ind_tr, ~] = datasample(female_ind, NumTrain/2, 'Replace', false);
    female_ind_te = setdiff(female_ind, female_ind_tr);
    ind_tr = sort([male_ind_tr female_ind_tr]);
    ind_te = sort([male_ind_te female_ind_te]);
    
    Xtr = X(:,ind_tr);
    Xte = X(:,ind_te);
    
    % generate indices of training data containing registration errors
    ind = randperm(NumTrain, round(rate_g * NumTrain)); 
    
    for i = 1:numROI
        fprintf('\n=============================================\n')
        fprintf('We are processing voxel i = %i (over %i)\n', i, numROI);
        
        % load clean training data and testing data
        file = sprintf(tensor_file_format2, rows(i), cols(i));
        load(file)
        Y = double( tensor(indx, :, :) );
        Y = permute(Y, [2:3,1]);
        Ytr_true = Y(:,:,ind_tr);
        Ytr = Ytr_true;
        Yte = Y(:,:,ind_te);        
        
        % load training data containing registration error
        file = sprintf(tensor_file_format1, rows(i), cols(i));
        load(file)
        Y = double( tensor(indx, :, :) );
        Y = permute(Y, [2:3,1]);
        Ytr(:,:,ind) = Y(:,:,ind); % only a fraction of training data 
                                   % contain registration error
        clear Y tensor
        
        % Multivariate geodesic regression with gross error
        [phat, Vhat, ~, Yhat, Yc, Ghat] = MGLM_Gross_spd(Xtr, Ytr, opts_wg);
        phat_wg(i,:,:) = phat;
        Vhat_wg(i,:,:,:) = Vhat;
        Yhat_wg(i,:,:,:) = Yhat;
        Yc_wg(i,:,:,:) = Yc;
        
        Ypred_wg = predSPD(phat, Vhat, Xte);
        PALMR_pred_error_roi(i,r) = MSGError_spd(Ypred_wg, Yte);
        
        % compute the precision and recall of identifying registration errors
        ngerr = 0; TP = 0; FN = 0;
        for t = 1:NumTrain
            if norm(Ghat(:,:,t),'fro') < 1e-8
                Ghat(:,:,t) = zeros(3);
            end
            v = Ghat(:,:,t); v = v(:);
            
            if any(v)
                ngerr = ngerr + 1;
            end
            
            if any(v) == 1 && ismember(t, ind)
                TP = TP + 1;
            end
            
            if any(v) == 0 && ismember(t, ind)
                FN = FN + 1;
            end
        end
        
        PALMR_G_precision(i,r) = TP / ngerr;
        PALMR_G_recall(i,r) = TP / (TP + FN);
        
        % Multivariate geodesic regression without gross error
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The original implementation of MGLM often results in
        % 'Numerical Error'. In such case, we can use palmr with 
        % specific parameters as an alternative choise, since MGLM 
        % is a specical case of PALMR when lambda = 0 and rho = inf.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        try
            [phat2, Vhat2, ~, Yhat2, ~] = mglm_spd(Xtr, Ytr);
        catch
            % alternative choice
            [phat2, Vhat2, ~, Yhat2, ~, ~] = MGLM_Gross_spd(Xtr, Ytr, opts_ng);
        end
        phat_ng(i,:,:) = phat2;
        Vhat_ng(i,:,:,:) = Vhat2;
        Yhat_ng(i,:,:,:) = Yhat2;
        
        Ypred_wg2 = predSPD(phat2, Vhat2, Xte);
        MGLM_pred_error_roi(i,r) = MSGError_spd(Ypred_wg2, Yte);
        
        % Fractional Anisotropy regression model
        FA_tr = Frac_Anisotropy(Ytr);
        FA_te = Frac_Anisotropy(Yte);
        [v, ~] = FA_MGLM_spd(Xtr, FA_tr);
        FA_pred = v * Xte;
        FA_pred_error_roi(i,r) = median( abs(FA_pred - FA_te) ./ FA_te );
        
        FA_palmr = Frac_Anisotropy(Ypred_wg);
        FA_mglm  = Frac_Anisotropy(Ypred_wg2);
        PALMR_FA_error_roi(i,r) = median( abs(FA_palmr - FA_te) ./ FA_te );
        MGLM_FA_error_roi(i,r) = median( abs(FA_mglm - FA_te) ./ FA_te );
        
        % print results
        fprintf('Prediction error:   Geodesic \t FA \n');
        fprintf('     PALMR model:   %.5f    %.5f \n', PALMR_pred_error_roi(i,r), ...
            PALMR_FA_error_roi(i,r));
        fprintf('      MGLM model:   %.5f    %.5f \n', MGLM_pred_error_roi(i,r), ...
            MGLM_FA_error_roi(i,r));
        fprintf('        FA model:     --       %.5f \n', FA_pred_error_roi(i,r));
        fprintf('Estimation of G\n\t No.Selected: %d\tPrecision: %.5f\tRecall: %.5f\n', ...
            ngerr, PALMR_G_precision(i,r), PALMR_G_recall(i,r));
    end
    save ../../results/Slice4_RegError_result.mat FA_pred_error_roi MGLM_pred_error_roi ...
        MGLM_FA_error_roi PALMR_FA_error_roi PALMR_pred_error_roi
end

%% print results
fprintf('\nPrediction error with registration error:\n')
fprintf('\tmedian FA error    median MSGE \n')
fprintf('FAreg:    %.5f \t\t -- \n', median(FA_pred_error_roi(:)))
fprintf('MGLM:    %.5f   \t   %.5f\n', median(MGLM_FA_error_roi(:)), ...
    median(MGLM_pred_error_roi(:)))
fprintf('PALMR:    %.5f   \t   %.5f\n', median(PALMR_FA_error_roi(:)), ...
    median(PALMR_pred_error_roi(:)))

