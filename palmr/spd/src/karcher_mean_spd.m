function Ybar = karcher_mean_spd(Y, w, maxit, tol)
% KARCHER_MEAN_SPD calculates the intrinsic mean with weight w on SPD manifolds.
%
%   Ybar = KARCHER_MEAN_SPD(Y, w, maxit, tol)
%   Ybar = KARCHER_MEAN_SPD(Y)
%
%       Y: a n-by-n-by-N array storing a set of N points on SPD(n) manifolds.
%       w: a column vector of lenght N storing weights, default value is all one vector.
%   maxit: the maximum iterations, default value is 500.
%     tol: tolerance of gradient descent iteration, default value is 1e-10; 
%    Ybar: a n-by-n SPD matrix which is the Karcher mean of Y.
%
% Written by Xiaowei Zhang 
% 2015/05/11
% updated on 2017/02/14

N = size(Y, 3);

if nargin < 4
    tol = 1e-10; 
    if nargin < 3
        maxit = 500;
        if nargin < 2
            w = ones(N,1) / N;
        end
    end    
end

% Set Ybar as a random point in Y as initialization
Ybar = Y(:,:,randi(N)); % or simply use Ybar = Y(:,:,1);

if length(w) ~= N
    error('The length of weight vector must be the same as the number of samples.');
end
w = w / norm(w,1);
for iter = 1:maxit
    phi = LogMapSPD(Ybar, Y);
    phi = weightedSum(phi, w); % negative gradient
    if Norm_TpM_spd(Ybar, phi) < tol  
        Ybar = ExpMapSPD(Ybar, phi);
        break
    end
    Ybar = ExpMapSPD(Ybar, phi);  % step size is 1
end