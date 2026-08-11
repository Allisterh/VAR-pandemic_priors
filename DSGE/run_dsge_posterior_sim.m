% RUN_DSGE_POSTERIOR_SIM  Posterior of the structural parameters on the
%                         simulated data, with vs without the Pandemic Priors,
%                         with the true parameter values overlaid.
%
%   The simulated-data counterpart of run_dsge_posterior.m. Because the data were
%   generated from known parameters, each parameter's true value is drawn as a
%   vertical line: the figure then shows that the with-dummy posterior brackets
%   the truth while the no-dummy posterior is biased away from it.
%
%   Reuses the exact simulated dataset, per-period phi*, warm-start modes and
%   measurement error saved by run_dsge_simulated.m in figures/simulated_results.mat.
%
%   Usage:  >> run_dsge_posterior_sim

function run_dsge_posterior_sim(ndraw, nburn)
if nargin<1 || isempty(ndraw), ndraw = 40000; end
if nargin<2 || isempty(nburn), nburn = 10000; end
here = fileparts(mfilename('fullpath')); cd(here);
addpath(fullfile(here,'utils'));                        % helper functions
set(0,'defaulttextinterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');
rng(20260715,'twister');

S = load(fullfile('figures','simulated_results.mat'), ...
         'y','pand','meas_sd','phi_star','th_nd','th_wd','th_true');
y = S.y;  pand = S.pand;  meas_sd = S.meas_sd;  phi_star = S.phi_star;
th_true = S.th_true;

optND = struct('use_dummy',false,'meas_sd',meas_sd);
optWD = struct('use_dummy',true,'pandemic',pand,'phi',phi_star,'meas_sd',meas_sd);

step0 = [0.06 0.05 0.05 0.05 0.06 0.06 0.06 0.06];

fprintf('\n=== MCMC posteriors (SIMULATED): phi* (per period)=[%s] | draws=%d (burn %d) ===\n', ...
        sprintf('%.3g ',phi_star), ndraw, nburn);
chain_nd = nk_mh(@(th) nk_logpost(th, y, optND), S.th_nd, step0, ndraw, nburn, 'no-dummy');
chain_wd = nk_mh(@(th) nk_logpost(th, y, optWD), S.th_wd, step0, ndraw, nburn, 'with-dummy');

% --- summary vs truth -----------------------------------------------------
plainnm = {'kappa','phi_pi','rho_a','rho_s','rho_m','sig_a','sig_s','sig_m'};
fprintf('\n%-9s | %8s | %19s | %19s\n','param','TRUE','no dummy (mean [5,95])','with dummy (mean [5,95])');
for k=1:8
    a=chain_nd(:,k); b=chain_wd(:,k);
    fprintf('%-9s | %8.3f | %6.3f [%6.3f,%6.3f] | %6.3f [%6.3f,%6.3f]\n', plainnm{k}, ...
        S.th_true(k), mean(a),prctile(a,5),prctile(a,95), mean(b),prctile(b,5),prctile(b,95));
end

% --- figure (shared; truth overlay) ---------------------------------------
outstem = fullfile('figures','simulated_posterior');
nk_posterior_fig(chain_nd, chain_wd, outstem, S.th_true);
save([outstem '.mat'], 'chain_nd','chain_wd','phi_star','th_true');
end
