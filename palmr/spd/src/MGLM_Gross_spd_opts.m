function opts = MGLM_Gross_spd_opts(opts)

% Options for MGLM_Gross_spd
%
% If opts is empty upon input, opts will be returned containing the default
% options for MGLM_Gross_spd.m.  
%
% Alternatively, if opts is passed with some fields already defined, those
% fields will be checked for errors, and the remaining fields will be added
% and initialized to their default values.
%
% Table of Options.  ** indicates default value.
%
% FIELD       DESCRIPTION
% .lambda     Regularization parameter of tangent vectors.
%             ** 0.1 **.
% .rho        Regularization parameter of gross error.
%             ** 0.5 **.
% .init       Specifies how p is to be initialized.  
%             ** 0 -> Karcher mean **
%             1 -> the first training data
%             2 -> a random training data
% .prox_para  Proximal parameter for variable p.
%             ** 1 **
% .tol        Tolerance on the change of two consecutive objective function values.
%             ** 1E-5 **
% .maxit      Maximum number of iterations.
%             ** 500 **
%
% Written by Xiaowei Zhang 
% 2015/05/13
% updated on 2017/02/14

if isfield(opts,'lambda')
    if opts.lambda < 0
        error('opts.lambda must be nonnegative.');
    end
else
    opts.lambda = 0.1;
end

if isfield(opts,'rho')
    if opts.rho < 0
        error('opts.rho1 must be nonnegative.');
    end
else
    opts.rho = 0.5;
end

if isfield(opts,'init')
    if ~isinInterval(opts.init,0,2,true) || opts.init ~= floor(opts.init)
        error('opts.init must be an integer between 0 and 2, inclusive.');
    end
else
    opts.init = 0; 
end

if isfield(opts,'prox_para')
    if opts.prox_para <= 0
        error('opts.prox_para must be positive.');
    end
else
    opts.prox_para = 1;
end

if isfield(opts,'tol')
    if (opts.tol <= 0) || (opts.tol >= 1)
        error('opts.tol is tolerance on objective function value decrement. Should be in (0,1).');
    end
else
    opts.tol = 1e-5;
end

if isfield(opts,'maxit')
    if (opts.maxit <= 0) || (opts.maxit ~= floor(opts.maxit)) 
        error('opts.maxit should be a positive integer.');
    end
else
    opts.maxit = 500;
end

if ~isfield(opts,'verbose'); 
    opts.verbose = 0; 
end

return