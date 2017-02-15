function [p, X, Y, Yrdn, Ytrue, V, G] = synth_spd_data(para)
% SYNTH_SPD_DATA generates the SPD data following a general linear model
%
%   [p, X, Y, Yrdn, Ytrue, V, G] = SYNTH_SPD_DATA(para)
%
%       p: a n-by-n SPD matrix (the intercept)
%       X: a nbasis-by-tnsample matrix storing the coefficients
%       Y: a n-by-n-by-tnsample array stroing SPD matrices
%    Yrdn: a n-by-n-by-tnsample array stroing SPD matrices obtained from Y
%          by adding stochastic noise
%   Ytrue: a n-by-n-by-tnsample array stroing SPD matrices without noise
%          or gross error
%       V: a n-by-n-by-nbasis array stotring symmetric matrices (basis tangent vectors)
%       G: a n-by-n-by-nbasis array stotring symmetric matrices (gross error tangent vectors)
%
% 
% Written by Xiaowei Zhang 
% 2015/05/18
% updated on 2017/02/14

if nargin < 1
    para = [];
end

% % get or check para
para = synth_spd_opts(para);

    dim = para.dim;      % dimension of the SPD matrices;
 nbasis = para.nbasis;   % number of basis tangent vectors
  noise = para.noise;    % stochastic noise level
 rate_g = para.rate_g;   % ratio of samples containing gross error 
noise_g = para.noise_g;  % noise level of gross error
  npair = para.npair;    % number of different pairs (x, y)
nsample = para.nsample;  % number of samples for each pair (x,y)

% Synthesized data
X = randn(nbasis,npair);
X = center(X);
V = zeros(dim, dim, nbasis);

X = repmat(X, 1, nsample);

while true
    p = randomSPD(dim);
    % Tangent vectors, geodesic bases.
    for j =1:nbasis
        y = randomSPD(dim);
        V(:,:,j) = LogMapSPD(p,y);
    end
    
    % Generate ground truth data
    Ytrue = zeros(dim, dim, npair);
    for i = 1:npair
        v = weightedSum(V, X(:,i)); % weighted sum of tangent vectors
        Ytrue(:,:,i) = ExpMapSPD(p, v);
    end
        
    % Stochastic noise
    tnsample = npair * nsample;  % total number of samples
    Yrdn = zeros(dim, dim, tnsample);
    
    index = 1;
    for i = 1:nsample
        for j = 1:npair
            Yrdn(:,:,index) = addNoiseSPD(Ytrue(:,:,j), noise);
            index = index + 1;
        end
    end
    Ytrue = repmat(Ytrue, [1 1 nsample]);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    % Gross error
    G = zeros(dim, dim, tnsample);
    Y = Yrdn;
    ind = randperm(tnsample);
    
    for j=1:ceil(tnsample * rate_g)
        [Y(:, :, ind(j)), G(:, :, ind(j))] = addGrossSPD(Yrdn(:, :, ind(j)), noise_g);
    end
    
    % check if all data are SPD matrices
    if SPDCheck(Ytrue) && SPDCheck(Yrdn) && SPDCheck(Y)
        break
    end

end % end of while

end % end of function