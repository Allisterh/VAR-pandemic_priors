function R = nk_fit_phi(y, pand, meas_sd, x0, opts)
% NK_FIT_PHI  Estimate the NK model with pandemic dummies, selecting one phi per
%             period by marginal likelihood (period-specific specification).
%
%   R = NK_FIT_PHI(y, pand, meas_sd, x0, opts)
%
%   Jointly maximizes over (theta, phi_1,...,phi_h):
%       obj = data-marginal-loglik(theta, phi)  +  nk_logprior(theta)
%   where h = numel(pand) is the number of pandemic periods (here 3: 2020Q1-Q3),
%   phi_h is the Gaussian-prior precision on that period's dummy coefficients
%   (shared across the n observables within the period), and the dummy
%   coefficients d are integrated out inside nk_kalman, so obj's phi-part is the
%   marginal likelihood and phi is selected by marginal likelihood (no prior on
%   phi), as in the BVAR. theta keeps its structural DSGE prior.
%
%   phi is optimized in log space (positive, wide range). theta uses the shared
%   nk_pack/nk_unpack transforms.
%
%   INPUTS
%     y        T-by-n data ; pand 1-by-h pandemic row indices
%     meas_sd  scalar or 1-by-n measurement-error s.d.
%     x0       8-vector start for theta in natural units (default plausible NK)
%     opts     optimset for fminsearch (optional)
%
%   OUTPUT struct R
%     .th       8-vector, estimated theta (natural units)
%     .phi      1-by-h, optimal per-period precisions
%     .loglik   data marginal log-likelihood at the optimum
%     .out      nk_kalman output at the optimum (smoothed states/shocks/d)
%     .h        number of pandemic periods
%
%   See also NK_KALMAN, NK_LOGPOST, NK_PROFILE_PHI.

h = numel(pand);
if nargin < 4 || isempty(x0),   x0 = [0.10 1.5 0.7 0.7 0.5 2.0 1.0 1.0]; end
if nargin < 5 || isempty(opts)
    opts = optimset('Display','off','MaxIter',2000,'MaxFunEvals',6000, ...
                    'TolX',1e-6,'TolFun',1e-6);
end

% pack: [theta(unconstrained 8) , log phi (h)]
z0  = [ nk_pack(x0) , log(0.1)*ones(1,h) ];      % start phi_h at 0.1
neg = @(z) -jointobj(z, y, pand, meas_sd, h);
zh  = fminsearch(neg, z0, opts);

th  = nk_unpack(zh(1:8));
phi = exp(zh(9:8+h));
opt = struct('use_dummy',true,'pandemic',pand,'phi',phi,'meas_sd',meas_sd);
[~, ll, out] = nk_logpost(th, y, opt);           % ll = data marginal loglik

R = struct('th',th,'phi',phi,'loglik',ll,'out',out,'h',h);
end

% ---------------------------------------------------------------------- %
function v = jointobj(z, y, pand, meas_sd, h)
% obj = data marginal loglik + structural prior on theta; phi has no prior term
% (so it is marginal-likelihood-selected). Returns -1e10 off-support.
    th  = nk_unpack(z(1:8));
    phi = exp(z(9:8+h));
    opt = struct('use_dummy',true,'pandemic',pand,'phi',phi,'meas_sd',meas_sd);
    lpost = nk_logpost(th, y, opt);              % = loglik + nk_logprior(theta)
    if ~isfinite(lpost), v = -1e10; else, v = lpost; end
end
