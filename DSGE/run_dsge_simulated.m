% RUN_DSGE_SIMULATED  Simulated-data validation of the DSGE pandemic-priors illustration.
%
%   Validates the mechanism on simulated data:
%     1. Simulate the NK model with normal shocks (known "truth").
%     2. Contaminate the 3 pandemic quarters (2020Q1-Q3) with a large wedge in
%        the observables (mimics the 2020 collapse/rebound).
%     3. Estimate the model without dummy handling  -> distorted params, monetary
%        IRF and smoothed monetary shock.
%     4. Estimate with the pandemic dummy, one phi per period selected by
%        marginal likelihood -> params/IRF/shocks ~ truth.
%     5. One figure: monetary IRF (x,pi) true vs no-dummy vs with-dummy, the
%        smoothed monetary shock over time, and the per-period profile marginal
%        likelihood curves (one per pandemic quarter) with phi_h* marked.
%
%   Engine: self-contained MATLAB (nk_solve/nk_irf/nk_kalman).
%
%   Usage:  >> run_dsge_simulated

clear; clc; close all;
here = fileparts(mfilename('fullpath')); cd(here);
addpath(fullfile(here,'utils'));                        % helper functions
set(0,'defaulttextinterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');
rng(20260715,'twister');                 % reproducible

% ------------------------------------------------------------------ setup
par0 = nk_params();                       % true parameters
sol0 = nk_solve(par0);
n    = 3;                                  % observables [x pi i]

% quarterly dates 1990Q1 ... 2023Q4  => pandemic 2020Q1-Q3
dates = (datetime(1990,1,1):calquarters(1):datetime(2023,10,1)).';
T     = numel(dates);
pand  = find( dates>=datetime(2020,1,1) & dates<=datetime(2020,7,1) ).';  % 3 rows
assert(numel(pand)==3, 'pandemic window must be 3 quarters');

Hirf   = 12;                               % IRF horizon (quarters), matches the US-data run
shk    = 'eps_m';                          % monetary shock
meas_sd = 1e-3;                            % tiny measurement error

% --------------------------------------------------- 1) simulate the truth
eps_true = (chol(sol0.Q,'lower') * randn(n,T)).';   % T-by-3 structural shocks
s = zeros(T,n);
for t = 2:T, s(t,:) = (sol0.F*s(t-1,:).').'; s(t,:) = s(t,:) + eps_true(t,:); end
s(1,:) = eps_true(1,:);
y_clean = (sol0.Z * s.').';                          % T-by-3 clean observables

% --------------------------------------------------- 2) contaminate 2020
% Large collapse-then-rebound wedge, biggest in 2020Q2, hitting all observables.
wedge = zeros(3,n);
%          x        pi       i
wedge(1,:) = [ -8.0,  -2.0,  -3.0 ];    % 2020Q1  demand collapse, rate cut
wedge(2,:) = [-25.0,  -5.0,  -6.0 ];    % 2020Q2  trough
wedge(3,:) = [ 12.0,   3.0,  -4.0 ];    % 2020Q3  rebound, rates still low
y = y_clean;
y(pand,:) = y(pand,:) + wedge;

% --------------------------------------------------- estimation setup
% Estimate the 8 parameters that shape the monetary IRF; fix beta,sigma,phi_x.
% theta (natural) = [kappa, phi_pi, rho_a, rho_s, rho_m, sig_a, sig_s, sig_m]
th_true = [par0.kappa, par0.phi_pi, par0.rho_a, par0.rho_s, par0.rho_m, ...
           par0.sig_a, par0.sig_s, par0.sig_m];
x0 = nk_pack(th_true);                     % start at truth (transformed)

opts = optimset('Display','off','MaxIter',800,'MaxFunEvals',4000, ...
                'TolX',1e-6,'TolFun',1e-6);

% --------------------------------------------------- 3) estimate without dummy
optND = struct('use_dummy',false,'meas_sd',meas_sd);
negll_nd = @(xx) -kalman_ll(nk_unpack(xx), y, optND);
xhat_nd  = fminsearch(negll_nd, x0, opts);
th_nd    = nk_unpack(xhat_nd);
[~, out_nd] = kalman_ll(th_nd, y, optND);
sol_nd   = nk_solve(nk_struct_par(th_nd));

% --------------------------------------------------- 4) estimate with dummy,
%     one phi per period selected by marginal likelihood (period-specific).
phi_grid = logspace(-3, 4, 25);            % small=drop obs ... large=ordinary
R = nk_fit_phi(y, pand, meas_sd, th_nd, opts);   % warm-start theta at no-dummy fit
th_wd    = R.th;  phi_star = R.phi;  out_wd = R.out;
sol_wd   = nk_solve(nk_struct_par(th_wd));
% per-period profile marginal-likelihood curves (figure input).
P = nk_profile_phi(R, y, pand, meas_sd, phi_grid);

% --------------------------------------------------- IRFs (monetary)
irf_true = nk_irf(sol0, shk, Hirf);
irf_nd   = nk_irf(sol_nd, shk, Hirf);
irf_wd   = nk_irf(sol_wd, shk, Hirf);
hz = (0:Hirf).';

% --------------------------------------------------- report table
names = {'kappa','phi_pi','rho_a','rho_s','rho_m','sig_a','sig_s','sig_m'};
fprintf('\n=== Simulated-data estimated parameters ===\n');
fprintf('%-8s %10s %12s %12s\n','param','TRUE','no-dummy','with-dummy');
for k = 1:numel(names)
    fprintf('%-8s %10.4f %12.4f %12.4f\n', names{k}, th_true(k), th_nd(k), th_wd(k));
end
fprintf('\noptimal phi per period (marginal-lik) = [%s]\n', sprintf('%.3g ',phi_star));
fprintf('smoothed monetary shock RMSE vs truth:  no-dummy = %.3f   with-dummy = %.3f\n', ...
        rmse(out_nd.eps_smooth(:,3), eps_true(:,3)), ...
        rmse(out_wd.eps_smooth(:,3), eps_true(:,3)));

% --------------------------------------------------- 5) figure
fig = figure('Position',[100 100 1100 760],'Color','w');
co = [0 0 0; 0.85 0.10 0.10; 0.00 0.45 0.85];   % true / no-dummy / with-dummy
lw = 1.8;

subplot(2,2,1); hold on; box on;
plot(hz, irf_true(:,1),'-','Color',co(1,:),'LineWidth',lw);
plot(hz, irf_nd(:,1),'--','Color',co(2,:),'LineWidth',lw);
plot(hz, irf_wd(:,1),'-.','Color',co(3,:),'LineWidth',lw);
yline(0,':','Color',[.5 .5 .5]);
title('Monetary IRF: output gap $x$'); xlabel('quarters'); ylabel('\% dev.');
xlim([0 Hirf]); xticks(0:4:Hirf);
legend({'true','no dummy','with dummy ($\phi^\ast$)'},'Location','best'); legend boxoff;

subplot(2,2,2); hold on; box on;
plot(hz, irf_true(:,2),'-','Color',co(1,:),'LineWidth',lw);
plot(hz, irf_nd(:,2),'--','Color',co(2,:),'LineWidth',lw);
plot(hz, irf_wd(:,2),'-.','Color',co(3,:),'LineWidth',lw);
yline(0,':','Color',[.5 .5 .5]);
title('Monetary IRF: inflation $\pi$'); xlabel('quarters'); ylabel('\% dev.');
xlim([0 Hirf]); xticks(0:4:Hirf);

subplot(2,2,3); hold on; box on;
zi = dates>=datetime(2017,1,1);                     % zoom around the pandemic
plot(dates(zi), eps_true(zi,3),'-','Color',co(1,:),'LineWidth',lw);
plot(dates(zi), out_nd.eps_smooth(zi,3),'--','Color',co(2,:),'LineWidth',lw);
plot(dates(zi), out_wd.eps_smooth(zi,3),'-.','Color',co(3,:),'LineWidth',lw);
yline(0,':','Color',[.5 .5 .5]);
xline(dates(pand(1)),':','Color',[.4 .4 .4]);
xline(dates(pand(end)),':','Color',[.4 .4 .4]);
title('Smoothed monetary shock $\varepsilon_m$'); ylabel('shock units');

subplot(2,2,4); hold on; box on;
% one profile marginal-likelihood curve per pandemic period, each
% relative to its own max (peak = that period's optimal phi_h).
pc = [0.85 0.10 0.10; 0.10 0.55 0.20; 0.00 0.45 0.85];  % Q1,Q2,Q3
qlab = {'2020Q1','2020Q2','2020Q3'};
hL = gobjects(1,P.h);
for hh = 1:P.h
    hL(hh) = semilogx(P.grid, P.llrel(hh,:),'-','Color',pc(hh,:),'LineWidth',lw);
    [~,ig] = max(P.llrel(hh,:));            % marker at each curve's own peak
    plot(P.grid(ig), P.llrel(hh,ig),'o','Color',pc(hh,:), ...
         'MarkerFaceColor',pc(hh,:),'MarkerSize',6);
end
set(gca,'XScale','log'); ylim([-15 1.5]);  % zoom to the peak; large-phi cliff runs off-frame
title('Per-period profile marg.\ likelihood ($\phi_h$)');
xlabel('$\phi_h$ (period-$h$ precision)'); ylabel('log marg.\ lik.\ (rel.\ max)');
legend(hL, qlab,'Location','best'); legend boxoff;

outpdf = fullfile('figures','simulated.pdf');
outpng = fullfile('figures','simulated.png');
exportgraphics(fig, outpdf, 'ContentType','vector');
exportgraphics(fig, outpng, 'Resolution',150);
fprintf('\nfigure saved: %s\n', outpdf);

save(fullfile('figures','simulated_results.mat'), ...
     'th_true','th_nd','th_wd','phi_star','phi_grid','P', ...
     'irf_true','irf_nd','irf_wd','eps_true','out_nd','out_wd', ...
     'dates','pand','meas_sd','y');   % y saved for the simulated MH posterior
fprintf('results saved: figures/simulated_results.mat\n');

% ====================================================================== %
%                          local functions                               %
% ====================================================================== %
% Shared model/prior/transform logic lives in standalone files: nk_struct_par,
% nk_logprior, nk_logpost, nk_pack, nk_unpack, nk_fit_phi, nk_profile_phi.
function [ll, out] = kalman_ll(th, y, opt)
% no-dummy objective, consistent with the US-data run: log-posterior kernel in theta
% (data marginal loglik + structural prior). Returns out.loglik = data marg lik.
    [obj, l, out] = nk_logpost(th, y, opt);
    ll = obj;
    if ~isfinite(ll), ll = -1e10; end
    if ~isfield(out,'loglik') || ~isfinite(l), out.loglik = -1e10; end
end
function r = rmse(a,b), m = isfinite(a)&isfinite(b); r = sqrt(mean((a(m)-b(m)).^2)); end
