% produce box plots in Figure 7 of the manuscript
%
% Written by Xiaowei Zhang 
% updated on 2017/02/14

clear all; close all; clc;
NumSlices = 6;

Improve_FA_NoErr = [];
Improve_FA_ManuErr = [];
Improve_FA_RegErr = [];

Improve_MSGE_NoErr = [];
Improve_MSGE_ManuErr = [];
Improve_MSGE_RegErr = [];

for i = 1:6
    load(sprintf('../../results/Slice%d_NoError_result.mat', i));
    load(sprintf('../../results/Slice%d_ManuError_result.mat', i));
    load(sprintf('../../results/Slice%d_RegError_result.mat', i));
    mask = logical(mask);
    
    % calculate median for each voxel
    MGLM_med_FA_err = median(MGLM_FA_error,3);
    MGLM_med_FA_err(isinf(MGLM_med_FA_err)) = 0;
    MGLM_med_FA_err(isnan(MGLM_med_FA_err)) = 0;
    PALMR_med_FA_err = median(PALMR_FA_error,3);
    PALMR_med_FA_err(isinf(PALMR_med_FA_err)) = 0;
    PALMR_med_FA_err(isnan(PALMR_med_FA_err)) = 0;
    
%     tmp = (median(MGLM_med_FA_err(mask)) - median(PALMR_med_FA_err(mask))) / median(MGLM_med_FA_err(mask)); 
%     Improve_FA_NoErr = [Improve_FA_NoErr tmp];

    tmp = (MGLM_med_FA_err(mask) - PALMR_med_FA_err(mask)) ./ MGLM_med_FA_err(mask); 
    Improve_FA_NoErr = [Improve_FA_NoErr; tmp];
    
    MGLM_med_MSGerr = median(MGLM_pred_error,3);
    PALMR_med_MSGerr = median(PALMR_pred_error,3);
    
%     tmp = (median(MGLM_med_MSGerr(mask)) - median(PALMR_med_MSGerr(mask))) / median(MGLM_med_MSGerr(mask)); 
%     Improve_MSGE_NoErr = [Improve_MSGE_NoErr tmp];
    
    tmp = (MGLM_med_MSGerr(mask) - PALMR_med_MSGerr(mask)) ./ MGLM_med_MSGerr(mask); 
    Improve_MSGE_NoErr = [Improve_MSGE_NoErr; tmp];
    
    MGLM_med_FA_err_noise = median(MGLM_FA_error_noise,3);
    PALMR_med_FA_err_noise = median(PALMR_FA_error_noise,3);
    
    tmp = (MGLM_med_FA_err_noise(mask) - PALMR_med_FA_err_noise(mask)) ./ ...
        MGLM_med_FA_err_noise(mask); 
    Improve_FA_ManuErr = [Improve_FA_ManuErr; tmp];
    
    MGLM_med_MSGerr_noise = median(MGLM_pred_error_noise,3);
    PALMR_med_MSGerr_noise = median(PALMR_pred_error_noise,3);
    
    tmp = (MGLM_med_MSGerr_noise(mask) - PALMR_med_MSGerr_noise(mask)) ./ ...
        MGLM_med_MSGerr_noise(mask); 
    Improve_MSGE_ManuErr = [Improve_MSGE_ManuErr; tmp];
    
    MGLM_med_FA_err_roi = median(MGLM_FA_error_roi);
    PALMR_med_FA_err_roi = median(PALMR_FA_error_roi);
    
    % row vector
    tmp = (MGLM_med_FA_err_roi - PALMR_med_FA_err_roi) ./ MGLM_med_FA_err_roi;
    Improve_FA_RegErr = [Improve_FA_RegErr tmp];
    
    MGLM_med_MSGerr_roi = median(MGLM_pred_error_roi);
    PALMR_med_MSGerr_roi = median(PALMR_pred_error_roi);
    
    % row vector
    tmp = (MGLM_med_MSGerr_roi - PALMR_med_MSGerr_roi) ./ MGLM_med_MSGerr_roi; 
    Improve_MSGE_RegErr = [Improve_MSGE_RegErr tmp];
    
    clear mask
end

figure(1)
err_val = 100 * [Improve_FA_NoErr; Improve_FA_ManuErr; Improve_FA_RegErr'];
xlbl = [zeros(length(Improve_FA_NoErr),1); ones(length(Improve_FA_ManuErr),1); 2 * ones(length(Improve_FA_RegErr), 1)];
h1 = boxplot(err_val, xlbl, 'width', 0.2, 'Labels', {'No gross error', '20% manual gross error', '20% registration error'});
set(h1,'LineWidth',2)
axis([0.7, 3.3, -100, 80])
grid on
ylabel('Improvement of relative FA error (%)');

figure(2)
err_val = 100 * [Improve_MSGE_NoErr; Improve_MSGE_ManuErr; Improve_MSGE_RegErr'];
xlbl = [zeros(length(Improve_MSGE_NoErr),1); ones(length(Improve_MSGE_ManuErr),1); 2 * ones(length(Improve_MSGE_RegErr), 1)];
h2 = boxplot(err_val, xlbl,  'width', 0.2, 'Labels', {'No gross error', '20% manual gross error', '20% registration error'});
set(h2,'LineWidth',2)
axis([0.7, 3.3, -60, 80])
grid on
ylabel('Improvement of MSGE (%)');