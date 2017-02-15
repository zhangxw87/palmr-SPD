% Print results and plot prediction errors on slice 3
%
% Written by Xiaowei Zhang 
% updated on 2017/02/15

clear all; close all; clc;
load('../../results/Slice3_NoError_result.mat');
load('../../results/Slice3_ManuError_result.mat');
load('../../results/Slice3_RegError_result.mat')
load('../../CMIND/data/wm_mask.mat')
mask = squeeze( mask(:, 64, :) );
mask = logical(mask);
NumVoxel = sum(mask(:));

%% calculate median for each voxel
FA_med_err = median(FA_pred_error,3);
MGLM_med_FA_err = median(MGLM_FA_error,3);
PALMR_med_FA_err = median(PALMR_FA_error,3);
MGLM_med_MSGerr = median(MGLM_pred_error,3);
PALMR_med_MSGerr = median(PALMR_pred_error,3);

FA_med_err_noise = median(FA_pred_error_noise,3);
MGLM_med_FA_err_noise = median(MGLM_FA_error_noise,3);
PALMR_med_FA_err_noise = median(PALMR_FA_error_noise,3);
MGLM_med_MSGerr_noise = median(MGLM_pred_error_noise,3);
PALMR_med_MSGerr_noise = median(PALMR_pred_error_noise,3);

FA_med_err_roi = median(FA_pred_error_roi(:));
MGLM_med_FA_err_roi = median(MGLM_FA_error_roi(:));
PALMR_med_FA_err_roi = median(PALMR_FA_error_roi(:));
MGLM_med_MSGerr_roi = median(MGLM_pred_error_roi(:));
PALMR_med_MSGerr_roi = median(PALMR_pred_error_roi(:));

fprintf('\nPrediction error without gross error:\n')
fprintf('\tmedian FA error    median MSGE \n')
fprintf('FAreg:    %.5f \t\t -- \n', median(FA_med_err(mask)))
fprintf('MGLM:    %.5f   \t   %.5f\n', median(MGLM_med_FA_err(mask)), median(MGLM_med_MSGerr(mask)))
fprintf('PALMR:    %.5f   \t   %.5f\n', median(PALMR_med_FA_err(mask)), median(PALMR_med_MSGerr(mask)))


fprintf('\nPrediction error with manual gross error:\n')
fprintf('\tmedian FA error    median MSGE \n')
fprintf('FAreg:    %.5f \t\t -- \n', median(FA_med_err_noise(mask)))
fprintf('MGLM:    %.5f   \t   %.5f\n', median(MGLM_med_FA_err_noise(mask)), median(MGLM_med_MSGerr_noise(mask)))
fprintf('PALMR:    %.5f   \t   %.5f\n', median(PALMR_med_FA_err_noise(mask)), median(PALMR_med_MSGerr_noise(mask)))

fprintf('\nPrediction error with registration error:\n')
fprintf('\tmedian FA error    median MSGE \n')
fprintf('FAreg:    %.5f \t\t -- \n', FA_med_err_roi)
fprintf('MGLM:    %.5f   \t   %.5f\n', MGLM_med_FA_err_roi, MGLM_med_MSGerr_roi)
fprintf('PALMR:    %.5f   \t   %.5f\n', PALMR_med_FA_err_roi, PALMR_med_MSGerr_roi)

% plot comarison between MGLM and PALMR
dev_fa = MGLM_med_FA_err - PALMR_med_FA_err;
ind_fa = abs(dev_fa) < 1e-3;
dev_fa(ind_fa) = 0;
dev_fa(dev_fa > 0) = 1;
dev_fa(dev_fa < 0) = -1;

dev = MGLM_med_MSGerr - PALMR_med_MSGerr;
ind = abs(dev) < 1e-3;
dev(ind) = 0;
dev(dev > 0) = 1;
dev(dev < 0) = -1;

dev_fa_noise = MGLM_med_FA_err_noise - PALMR_med_FA_err_noise;
ind_fa_noise = abs(dev_fa_noise) < 1e-3;
dev_fa_noise(ind_fa_noise) = 0;
dev_fa_noise(dev_fa_noise > 0) = 1;
dev_fa_noise(dev_fa_noise < 0) = -1;

dev_noise = MGLM_med_MSGerr_noise - PALMR_med_MSGerr_noise;
ind_noise = abs(dev_noise) < 1e-3;
dev_fa(ind_noise) = 0;
dev_noise(dev_noise > 0) = 1;
dev_noise(dev_noise < 0) = -1;

figure(1)
subplot(2,2,1)
colormap('gray')
imagesc(rot90(dev_fa, 1))
axis tight
hcb=colorbar;
set(hcb,'YTick', [-1, 0, 1])
xlabel('(a)')

subplot(2,2,2)
colormap('gray')
imagesc(rot90(dev, 1))
axis tight
hcb=colorbar;
set(hcb,'YTick', [-1, 0, 1])
xlabel('(b)')

subplot(2,2,3)
colormap('gray')
imagesc(rot90(dev_fa_noise, 1))
axis tight
hcb=colorbar;
set(hcb,'YTick', [-1, 0, 1])
xlabel('(c)')

subplot(2,2,4)
colormap('gray')
imagesc(rot90(dev_noise, 1))
axis tight
hcb=colorbar;
set(hcb,'YTick', [-1, 0, 1])
xlabel('(d)')

%% FA models
Num_bin = 50;
temp1 = FA_med_err(mask);
temp1(isinf(temp1)) = [];
temp1(isnan(temp1)) = [];
temp1 = temp1(temp1 < 10);
[fa_nele, fa_cen] = hist(temp1,Num_bin);

temp2 = FA_med_err_noise(mask);
temp2(isinf(temp2)) = [];
temp2(isnan(temp2)) = [];
temp2 = temp2(temp2 < 10);
[fa_nele_noise, fa_cen_noise] = hist(temp2,Num_bin);

temp3 = PALMR_med_FA_err(mask);
temp3(isinf(temp3)) = [];
temp3(isnan(temp3)) = [];
temp3 = temp3(temp3 < 10);
[palmr_fa_nele, palmr_fa_cen] = hist(temp3,Num_bin);

temp4 = PALMR_med_FA_err_noise(mask);
temp4(isinf(temp4)) = [];
temp4(isnan(temp4)) = [];
temp4 = temp4(temp4 < 10);
[palmr_fa_nele_noise, palmr_fa_cen_noise] = hist(temp4,Num_bin);

temp5 = MGLM_med_FA_err(mask);
temp5(isinf(temp5)) = [];
temp5(isnan(temp5)) = [];
temp5 = temp5(temp5 < 10);
[mglm_fa_nele, mglm_fa_cen] = hist(temp5,Num_bin);

temp6 = MGLM_med_FA_err_noise(mask);
temp6(isinf(temp6)) = [];
temp6(isnan(temp6)) = [];
temp6 = temp6(temp6 < 10);
[mglm_fa_nele_noise, mglm_fa_cen_noise] = hist(temp6,Num_bin);

fig2 = figure(2);
plot(palmr_fa_cen, palmr_fa_nele / NumVoxel, 'b-', mglm_fa_cen, mglm_fa_nele / NumVoxel, 'r--', ...
    fa_cen, fa_nele / NumVoxel, 'g-.', palmr_fa_cen_noise, palmr_fa_nele_noise / NumVoxel, 'k-', ...
    'LineWidth',2);
hold on;
plot(mglm_fa_cen_noise, mglm_fa_nele_noise / NumVoxel, '--', 'color', [0 0.6 0.6], 'LineWidth',2);
plot(fa_cen_noise, fa_nele_noise / NumVoxel, '-.', 'color', [1 0.5 0], 'LineWidth',2);
legend('PALMR', 'MGLM', 'FA regression', 'PALMR with 20% gross error', 'MGLM with 20% gross error', 'FA regression with 20% gross error');
xlabel('Relative FA error', 'FontSize', 18);
ylabel('Empirical probability', 'FontSize', 18);
axis([0, 5, -0.01, 0.4]);

fig3 = figure(3);
plot(palmr_fa_cen, palmr_fa_nele / NumVoxel, 'b-', mglm_fa_cen, mglm_fa_nele / NumVoxel, 'r--', ...
    palmr_fa_cen_noise, palmr_fa_nele_noise / NumVoxel, 'k-', 'LineWidth',2);
hold on;
plot(mglm_fa_cen_noise, mglm_fa_nele_noise / NumVoxel, '--', 'color', [0 0.6 0.6], 'LineWidth',2);
axis([0, 1, -0.01, 0.3]);

new_fig = figure(4);
inset_size = 0.41;
main_fig = findobj(fig2,'Type','axes');
h_main = copyobj(main_fig,new_fig);
main_fig_pos = get(main_fig,'Position');
set(h_main(2),'Position',[0.13, 0.11, 0.775, 0.815]);
inset_fig = findobj(fig3,'Type','axes');
h_inset = copyobj(inset_fig,new_fig);
ax = main_fig_pos{2};
set(h_inset,'Position', [0.48 0.25 inset_size inset_size]);

close(fig2)
close(fig3)

%% manifold models with manual gross error
Num_bin2 = 200;
temp1 = MGLM_med_MSGerr(mask);
temp1(isinf(temp1)) = [];
temp1(isnan(temp1)) = [];
temp1 = temp1(temp1 < 10);
[mglm_nele, mglm_cen] = hist(temp1,Num_bin2);

temp2 = PALMR_med_MSGerr(mask);
temp2(isinf(temp2)) = [];
temp2(isnan(temp2)) = [];
temp2 = temp2(temp2 < 10);
[palmr_nele, palmr_cen] = hist(temp2,Num_bin2);

temp3 = MGLM_med_MSGerr_noise(mask);
temp3(isinf(temp3)) = [];
temp3(isnan(temp3)) = [];
temp3 = temp3(temp3 < 10);
[mglm_nele_noise, mglm_cen_noise] = hist(temp3,Num_bin2);

temp4 = PALMR_med_MSGerr_noise(mask);
temp4(isinf(temp4)) = [];
temp4(isnan(temp4)) = [];
temp4 = temp4(temp4 < 10);
[palmr_nele_noise, palmr_cen_noise] = hist(temp4,Num_bin2);

figure(5)
plot(palmr_cen, palmr_nele / NumVoxel, 'b-', mglm_cen, mglm_nele / NumVoxel, 'r--', ...
    palmr_cen_noise, palmr_nele_noise / NumVoxel, 'k-.', 'LineWidth',2);
hold on;
plot(mglm_cen_noise, mglm_nele_noise / NumVoxel, ':', 'color', [0 0.6 0.6], 'LineWidth',4);
legend('PALMR', 'MGLM', 'PALMR with 20% gross error', 'MGLM with 20% gross error')
xlabel('MSGE', 'FontSize', 18);
ylabel('Empirical probability', 'FontSize', 18);
axis([0, 1, -0.01, 0.25]);
set(gca,'XTick', 0:0.2:1)