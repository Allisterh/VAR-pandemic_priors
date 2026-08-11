function P = nk_profile_phi(R, y, pand, meas_sd, phi_grid)
% NK_PROFILE_PHI  Per-period profile marginal-likelihood curves.
%
%   P = NK_PROFILE_PHI(R, y, pand, meas_sd, phi_grid)
%
%   For each pandemic period h, varies phi_h across phi_grid while holding the
%   other periods' phi fixed at their joint optimum R.phi and theta fixed at the
%   joint optimum R.th (a true profile slice through the optimum). Returns the
%   data marginal log-likelihood along each slice. The simulated- and US-data figures
%   plot these as h lines (one per pandemic quarter), each relative to its own
%   maximum so curvature is comparable: a sharp peak at small phi => that quarter
%   is strongly downweighted; a flat curve => that quarter matters little. The
%   peak of curve h sits at R.phi(h) by construction.
%
%   INPUTS
%     R        output of nk_fit_phi (uses R.th, R.phi, R.h)
%     y,pand,meas_sd  as in nk_fit_phi
%     phi_grid 1-by-G grid of phi values (default logspace(-3,4,25))
%
%   OUTPUT struct P
%     .grid    1-by-G phi grid
%     .ll      h-by-G raw profile marginal log-likelihood
%     .llrel   h-by-G profile ll minus each row's own max (peak = 0)
%     .phi_opt 1-by-h the joint-optimal phi (= R.phi), for peak markers

if nargin < 5 || isempty(phi_grid), phi_grid = logspace(-3, 4, 25); end
h = R.h;  G = numel(phi_grid);
ll = nan(h, G);

for hh = 1:h
    for g = 1:G
        phi = R.phi;  phi(hh) = phi_grid(g);         % vary only period hh
        opt = struct('use_dummy',true,'pandemic',pand,'phi',phi,'meas_sd',meas_sd);
        [~, l] = nk_logpost(R.th, y, opt);           % data marginal loglik
        ll(hh,g) = l;
    end
end
ll(~isfinite(ll)) = NaN;

P.grid    = phi_grid;
P.ll      = ll;
P.llrel   = ll - max(ll,[],2,'omitnan');
P.phi_opt = R.phi;
P.h       = h;
end
