%% generate figures in the illustrative example of the paper
%
% Written by Xiaowei Zhang 
% 2017/02/14

clear; clc; close all;

addpath('../../../../riem_mglm_0.12/spd');
addpath('../../../../riem_mglm_0.12/common');
addpath('../../src');

dim = 3;        % dimension of spd matrix
nbasis = 2;     % number of basis
npair = 10;     % number of different values {x_i}
nsample = 5;    % number of training samples for each {x_i}
noise = 0.1;    % stochastic noise magnitude
noise_g = 5;    % gross error magnitude
rate_g = 0.4;   % ratio of training date with gross error
nval = 20;      % number of validation data
ntest = 20;     % number of testing data

% independent variable X of size nbasis-by-npair
rng(0);
X = [0:0.25:1 0:0.25:1; 0 0 0 0 0 1 1 1 1 1 ];
p = randomSPD(dim); % base point

% tangent vectors, geodesic bases.
V = zeros(dim,dim,nbasis);
for j =1:nbasis
    y = randomSPD(dim);
    V(:,:,j) = LogMapSPD(p,y);
end

% generate ground truth data
Ytrue = zeros(dim, dim, npair);
for i = 1:npair
    v = weightedSum(V, X(:,i)); % weighted sum of tangent vectors
    Ytrue(:,:,i) = ExpMapSPD(p, v);
end

% stochastic noise
tnsample = npair * nsample;  % total number of samples
Yrdn = zeros(dim, dim, tnsample); % training samples with only random noise

index = 1;
for i = 1:nsample
    for j = 1:npair
        Yrdn(:,:,index) = addNoiseSPD(Ytrue(:,:,j), noise);
        index = index + 1;
    end
end
Ytrue = repmat(Ytrue, [1 1 nsample]); % ground-truth training samples 
X = repmat(X, 1, nsample); % independent variables {x_i}

% gross error
G = zeros(dim, dim, tnsample);
Y = Yrdn;
ind = randperm(tnsample);
num_gross = ceil(tnsample * rate_g);
% inject gross error into the training data
for j = 1:num_gross
    [Y(:, :, ind(j)), G(:, :, ind(j))] = addGrossSPD(Yrdn(:, :, ind(j)), noise_g);
end

% generate testing data
Xval = randn(nbasis, nval);
Yval = predSPD(p, V, Xval);

% generate testing data
Xte = randn(nbasis, ntest);
Yte = predSPD(p, V, Xte);

%% Multivariate geodesic regression with gross error

% regularization parameters
lambdaSet = 0.1;
rhoSet = 0.05:0.05:1;

t1 = tic;
% use validation set to select the optimal model parameter
opts.prox_para = 50;
opts.maxit = 50; % maximum number of iterations for cross validation
[row, col] = MGLM_Gross_spd_cv(X, Y, Xval, Yval, lambdaSet, rhoSet, opts);
fprintf('The optimal regularization parameters: lambda = %f, rho = %f. \n',...
    lambdaSet(row), rhoSet(col));
opts.lambda = lambdaSet(row);
opts.rho = rhoSet(col);

opts.verbose = 1; % change this value to 0, if you do not want to show intermediate results
opts.maxit = 100; % maximum number of iterations
fprintf('Start of geodesic regression with gross error:\n');
[phat_wg, Vhat_wg, ~, Yhat_wg, Yc_wg, Ghat_wg] = MGLM_Gross_spd(X, Y, opts);
OutputWithG.Time = toc(t1);
Yte_wg = predSPD(phat_wg, Vhat_wg, Xte);

% calculate errors
OutputWithG.MSGE_Ytrain = MSGError_spd(Ytrue, Yhat_wg);
OutputWithG.MSGE_Ytest = MSGError_spd(Yte, Yte_wg);
OutputWithG.MSGE_Yc = MSGError_spd(Yrdn, Yc_wg);
OutputWithG.MSGE_p = MSGError_spd(p, phat_wg);
OutputWithG.MSGE_V = MSGError_TpM_spd(p, V, phat_wg, Vhat_wg);

Ghat_wg2 = zeros(size(G));
OutputWithG.MSGE_G = 0;
OutputWithG.MSGE_Gratio = 0;
for j = 1:size(G,3)
    u = G(:,:,j); u = u(:);
    v = Ghat_wg(:,:,j); v = v(:);
    
    if any(v)
        Ghat_wg2(:,:,j) = LogMapSPD(Yc_wg(:,:,j), Y(:,:,j));
    end
    OutputWithG.MSGE_G = OutputWithG.MSGE_G + ...
        MSGError_TpM_spd(Yrdn(:,:,j), G(:,:,j), Yc_wg(:,:,j), Ghat_wg2(:,:,j));
    
    if any(u) == any(v)
        OutputWithG.MSGE_Gratio = OutputWithG.MSGE_Gratio + 1;
    end
end
OutputWithG.MSGE_G = OutputWithG.MSGE_G / size(G,3);
OutputWithG.MSGE_Gratio = OutputWithG.MSGE_Gratio / size(G,3);

%% Multivariate geodesic regression without gross error
fprintf('Start of geodesic regression w/o gross error:\n')

t2 = tic;
[phat_ng, Vhat_ng, ~, Yhat_ng, ~] = mglm_spd(X,Y);
OutputNoG.Time = toc(t2);
Yte_ng = predSPD(phat_ng, Vhat_ng, Xte);

OutputNoG.MSGE_Ytrain = MSGError_spd(Ytrue, Yhat_ng);
OutputNoG.MSGE_Ytest = MSGError_spd(Yte, Yte_ng);
OutputNoG.MSGE_p = MSGError_spd(p, phat_ng);
OutputNoG.MSGE_V = MSGError_TpM_spd(p, V, phat_ng, Vhat_ng);

%% print results
fprintf('   Mean Squared Geodesic Errors:     PALMR  \t  MGLM \n');
fprintf('         Error on training data:   %.5f  \t %.5f \n', ...
    OutputWithG.MSGE_Ytrain, OutputNoG.MSGE_Ytrain);
fprintf('          Error on testing data:   %.5f  \t %.5f \n', ...
    OutputWithG.MSGE_Ytest, OutputNoG.MSGE_Ytest);
fprintf('          Error of estimating p:   %.5f  \t %.5f \n', ...
    OutputWithG.MSGE_p, OutputNoG.MSGE_p);
fprintf('          Error of estimating V:   %.5f  \t %.5f \n', ...
    OutputWithG.MSGE_V, OutputNoG.MSGE_V);
fprintf('            Error of correction:   %.5f  \t   --  \n', OutputWithG.MSGE_Yc);
fprintf('          Error of estimating G:   %.5f  \t   --  \n', OutputWithG.MSGE_G);
fprintf('Ratio of correctly identified G:   %.5f  \t   --  \n', OutputWithG.MSGE_Gratio);

%% plot figures
f1 = figure(1); 
Temp = zeros([dim, dim, npair, 3]);
Temp(:,:,:,1) = Ytrue(:,:,1:npair); % ground truth training 
Temp(:,:,:,2) = Yhat_wg(:,:,1:npair); % PALMR prediction on training data
Temp(:,:,:,3) = Yhat_ng(:,:,1:npair); % MGLM prediction on training data
plotDTI(Temp); clear Temp;
set(f1,'NextPlot','add');
axes;
h = title('Prediction on training data', 'FontSize', 15);
set(gca,'Visible','off');
set(h,'Visible','on');


f2 = figure(2); % all training samples
plotDTI(reshape(Y, [dim, dim, npair, nsample]))
set(f2,'NextPlot','add');
axes;
h = title('All training samples', 'FontSize', 15);
set(gca,'Visible','off');
set(h,'Visible','on');


f3 = figure(3); % plot testing data
Temp = zeros([dim, dim, ntest, 3]);
Temp(:,:,:,1) = Yte; % ground truth testing
Temp(:,:,:,2) = Yte_wg; % PALMR prediction on testing data
Temp(:,:,:,3) = Yte_ng; % MGLM prediction on testing data
plotDTI(Temp); clear Temp;
set(f3,'NextPlot','add');
axes;
h = title('Prediction on testing data', 'FontSize', 15);
set(gca,'Visible','off');
set(h,'Visible','on');


f4 = figure(4); % plot corrupted training data
Temp = zeros([dim, dim, num_gross, 3]);
Temp(:,:,:,1) = Y(:,:,ind(1:num_gross)); % training data with corruption
Temp(:,:,:,2) = Yc_wg(:,:,ind(1:num_gross)); % PALMR correction
Temp(:,:,:,3) = Yrdn(:,:,ind(1:num_gross)); % training data without corruption
plotDTI(Temp); clear Temp;
set(f4,'NextPlot','add');
axes;
h = title('Correction on corrupted training data', 'FontSize', 15);
set(gca,'Visible','off');
set(h,'Visible','on');