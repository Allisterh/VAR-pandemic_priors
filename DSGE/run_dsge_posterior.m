% RUN_DSGE_POSTERIOR  Posterior distributions of the structural parameters on US
%                     data, with vs without the Pandemic Priors (the DSGE analogue
%                     of the BVAR coefficient-distribution figure).
%
%   Draws the posterior of the 8 structural parameters theta by random-walk
%   Metropolis-Hastings (nk_mh), in two specifications:
%     - no dummy   : ordinary Bayesian DSGE estimation on the full sample, the
%                    pandemic quarters treated as normal data.
%     - with dummy : pandemic dummy coefficients d in the measurement equation at
%                    2020Q1-Q3 with the per-period Pandemic Prior (precision phi_h),
%                    integrated out inside nk_kalman, phi fixed at the per-period
%                    marginal-likelihood optimum from run_dsge_usdata.m.
%   Both target nk_logpost (data marg. lik + structural prior); the only
%   difference is the pandemic dummies.
%
%   Usage:  >> run_dsge_posterior                    % default headline combo
%           >> run_dsge_posterior('GDPDEF_FEDFUNDS') % a saved US-data combo
%
%   Requires figures/usdata_<combo>.mat (phi_star vector, modes) from run_dsge_usdata.m.

function run_dsge_posterior(tag, ndraw, nburn)
if nargin<1 || isempty(tag),   tag   = 'PCEPILFE_FEDFUNDS'; end
if nargin<2 || isempty(ndraw), ndraw = 40000; end
if nargin<3 || isempty(nburn), nburn = 10000; end
here = fileparts(mfilename('fullpath')); cd(here);
addpath(fullfile(here,'utils'));                        % helper functions
set(0,'defaulttextinterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');
rng(20260715,'twister');

% --- recover data + settings from the US-data run -------------------------
parts = split(string(tag),'_');
infl  = char(parts(1));
rate  = char(strjoin(parts(2:end),'_'));
D = load_nk_data(infl, rate);
y = D.y;  pand = D.pandemic;  dates = D.dates;
nonp = true(size(dates)); nonp(pand)=false;
meas_sd = 0.05 * std(y(nonp,:),0,1);          % same rule as run_dsge_usdata.m

S = load(fullfile('figures',['usdata_' tag '.mat']));
phi_star = S.phi_star;                         % per-period vector
th_nd0 = S.th_nd;  th_wd0 = S.th_wd;           % modes (warm starts)

optND = struct('use_dummy',false,'meas_sd',meas_sd);
optWD = struct('use_dummy',true,'pandemic',pand,'phi',phi_star,'meas_sd',meas_sd);

step0 = [0.06 0.05 0.05 0.05 0.06 0.06 0.06 0.06];

fprintf('\n=== MCMC posteriors (%s): phi* (per period)=[%s] | draws=%d (burn %d) ===\n', ...
        nk_dispname(tag), sprintf('%.3g ',phi_star), ndraw, nburn);
chain_nd = nk_mh(@(th) nk_logpost(th, y, optND), th_nd0, step0, ndraw, nburn, 'no-dummy');
chain_wd = nk_mh(@(th) nk_logpost(th, y, optWD), th_wd0, step0, ndraw, nburn, 'with-dummy');

% --- posterior summary table ---------------------------------------------
fprintf('\n%-9s | %19s | %19s\n','param','no dummy (mean [5,95])','with dummy (mean [5,95])');
plainnm = {'kappa','phi_pi','rho_a','rho_s','rho_m','sig_a','sig_s','sig_m'};
for k=1:8
    a=chain_nd(:,k); b=chain_wd(:,k);
    fprintf('%-9s | %6.3f [%6.3f,%6.3f] | %6.3f [%6.3f,%6.3f]\n', plainnm{k}, ...
        mean(a),prctile(a,5),prctile(a,95), mean(b),prctile(b,5),prctile(b,95));
end

% --- figure (shared; no truth overlay for US data) ------------------------
outstem = fullfile('figures',['usdata_posterior_' tag]);
nk_posterior_fig(chain_nd, chain_wd, outstem);
save([outstem '.mat'], 'chain_nd','chain_wd','phi_star','tag');
end
