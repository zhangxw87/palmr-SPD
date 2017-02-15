function para = synth_spd_opts(para)

% para for generating spd matrices
%
% If para is empty upon input, para will be returned containing the default
% options for synth_spd_data.m.  
%
% Alternatively, if para is passed with some fields already defined, those
% fields will be checked for errors, and the remaining fields will be added
% and initialized to their default values.
%
% Table of Options.  ** indicates default value.
%
% FIELD      DESCRIPTION
% .dim       Dimension of the spd matrices.
%            ** 3 **.
% .nbasis    Number of basis tangent vectors.
%            ** 2 **.
% .npair     Number of pairs of data (x,y).  
%            ** 10 **
% .nsample   Number of sample for each pair of (x, y).
%            ** 5 **
% .noise     Stochastic noise magnitude.
%            ** 0.1 **
% .noise_g   Gross error magnitude.
%            ** 1 **
% .rate_g    The ratio of data containing gross error.
%            ** 0.2 **
%
% 
% Written by Xiaowei Zhang 
% 2015/05/18
% updated on 2017/02/14

if isfield(para,'dim')
    if (para.dim <= 0) || (para.dim ~= floor(para.dim)) 
        error('para.dim should be a positive integer.');
    end
else
    para.dim = 3;
end

if isfield(para,'nbasis')
    if (para.nbasis <= 0) || (para.nbasis ~= floor(para.nbasis)) 
        error('para.nbasis should be a positive integer.');
    end
else
    para.nbasis = 2;
end

if isfield(para,'npair')
    if (para.npair <= 0) || (para.npair ~= floor(para.npair)) 
        error('para.npair should be a positive integer.');
    end
else
    para.npair = 10;
end

if isfield(para,'nsample')
    if (para.nsample <= 0) || (para.nsample ~= floor(para.nsample)) 
        error('para.nsample should be a positive integer.');
    end
else
    para.nsample = 5;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
end

if isfield(para,'noise')
    if para.noise < 0
        error('para.noise must be nonnegative.');
    end
else
    para.noise = 0.1;
end

if isfield(para,'noise_g')
    if para.noise_g < 0
        error('para.noise_g must be nonnegative.');
    end
else
    para.noise_g = 1;
end

if isfield(para,'rate_g')
    if (para.rate_g <= 0) || (para.rate_g > 1)
        error('para.rate_g is the ratio of data containing gross error. Should be in (0,1).');
    end
else
    para.rate_g = 0.2;
end

return