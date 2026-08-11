% RUN_DSGE_USDATA  US-data application of the DSGE pandemic-priors illustration.
%
%   Estimates the 3-equation NK model on US data with and without the pandemic
%   dummy, phi selected by marginal likelihood, for each inflation x policy-rate
%   combination, and saves one figure per combo plus a summary. Mirrors the
%   simulated-data run (run_dsge_simulated.m); here there is no known "truth", so
%   the illustration is:
%     - without dummy: the 2020 collapse/rebound is forced into the structural
%       monetary shock and distorts the estimated monetary IRF.
%     - with dummy (phi*): the pandemic wedge d absorbs 2020 in the measurement
%       equation, so the monetary shock and IRF return to sensible magnitudes.
%
%   Engine: self-contained MATLAB (nk_solve/nk_irf/nk_kalman); phi selected by
%   maximizing the marginal likelihood (dummy coefficients integrated out).
%
%   Usage:  >> run_dsge_usdata

clear; clc; close all;
here = fileparts(mfilename('fullpath')); cd(here);
addpath(fullfile(here,'utils'));                        % helper functions
set(0,'defaulttextinterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');

infl_opts = {'GDPDEF','PCEPILFE'};
rate_opts = {'SHADOW_SPLICED','FEDFUNDS'};
Hirf   = 12;  shk = 'eps_m';
% Small measurement error per observable (~5% of each series' s.d.), set inside
% the combo loop from the data. Keeps the 3-obs/3-shock filter away from exact
% singularity without materially absorbing signal.
meas_frac = 0.05;
phi_grid = logspace(-3, 4, 22);

opts = optimset('Display','off','MaxIter',1000,'MaxFunEvals',5000, ...
                'TolX',1e-6,'TolFun',1e-6);

summary = {};
for ii = 1:numel(infl_opts)
for jj = 1:numel(rate_opts)
    infl = infl_opts{ii};  rate = rate_opts{jj};
    D = load_nk_data(infl, rate);
    y = D.y;  pand = D.pandemic;  dates = D.dates;

    % per-observable measurement error (~5% of each series' s.d.),
    % excluding the pandemic quarters from the s.d. so 2020 does not inflate it
    nonp = true(size(dates)); nonp(pand) = false;
    meas_sd = meas_frac * std(y(nonp,:), 0, 1);   % 1-by-3

    % start values (natural): plausible NK calibration
    th0 = [0.10, 1.5, 0.7, 0.7, 0.5, 2.0, 1.0, 1.0];
    x0  = pack(th0);

    % ---- without dummy ----
    optND = struct('use_dummy',false,'meas_sd',meas_sd);
    negND = @(xx) -kalman_ll(unpack(xx), y, optND);
    xnd   = fminsearch(negND, x0, opts);
    th_nd = unpack(xnd);
    [~,out_nd] = kalman_ll(th_nd, y, optND);
    sol_nd = nk_solve(nk_struct_par(th_nd));

    % ---- with dummy: jointly optimize (theta, phi_1..phi_h) by marginal lik ----
    % one phi per period (period-specific). Dummy coeffs integrated out, so phi is
    % marginal-likelihood-selected; theta keeps its structural prior.
    R = nk_fit_phi(y, pand, meas_sd, th_nd, opts);   % warm-start theta at no-dummy fit
    th_wd = R.th;  phi_star = R.phi;  out_wd = R.out;
    sol_wd = nk_solve(nk_struct_par(th_wd));
    % per-period profile marginal-likelihood curves (figure input).
    P = nk_profile_phi(R, y, pand, meas_sd, phi_grid);

    % ---- IRFs (monetary) ----
    irf_nd = nk_irf(sol_nd, shk, Hirf);
    irf_wd = nk_irf(sol_wd, shk, Hirf);
    hz = (0:Hirf).';

    % ---- report ----
    tag = sprintf('%s_%s', infl, rate);
    nm  = {'kappa','phi_pi','rho_a','rho_s','rho_m','sig_a','sig_s','sig_m'};
    fprintf('\n===== %s | %s =====\n', infl, rate);
    fprintf('%-8s %12s %12s\n','param','no-dummy','with-dummy');
    for k=1:8, fprintf('%-8s %12.4f %12.4f\n', nm{k}, th_nd(k), th_wd(k)); end
    % peak (impact) monetary IRF on output gap: distortion measure
    peak_nd = irf_nd(1,1);  peak_wd = irf_wd(1,1);
    % max |smoothed monetary shock| in 2020 vs pre-2020 sd
    em = out_nd.eps_smooth(:,3);
    pre = dates<datetime(2020,1,1);
    ratio_nd = max(abs(em(pand))) / std(em(pre),'omitnan');
    emw = out_wd.eps_smooth(:,3);
    ratio_wd = max(abs(emw(pand))) / std(emw(pre),'omitnan');
    fprintf('phi* (per period) = [%s] | impact IRF(x): no-dummy %.3f, with-dummy %.3f\n', ...
            sprintf('%.3g ',phi_star), peak_nd, peak_wd);
    fprintf('2020 monetary shock spike / pre-2020 sd: no-dummy %.1f, with-dummy %.1f\n', ...
            ratio_nd, ratio_wd);
    summary(end+1,:) = {tag, phi_star(1), phi_star(2), phi_star(3), ...
                        peak_nd, peak_wd, ratio_nd, ratio_wd}; %#ok<SAGROW>

    % ---- figure ----
    fig = figure('Position',[80 80 1150 780],'Color','w','Visible','off');
    co = [0.85 0.10 0.10; 0.00 0.45 0.85];  lw = 1.8;

    subplot(2,3,1); hold on; box on;
    plot(hz, irf_nd(:,1),'--','Color',co(1,:),'LineWidth',lw);
    plot(hz, irf_wd(:,1),'-.','Color',co(2,:),'LineWidth',lw);
    yline(0,':',Color=[.5 .5 .5]); title('Monetary IRF: output gap $x$');
    xlabel('quarters'); ylabel('\% dev.'); xlim([0 Hirf]); xticks(0:4:Hirf);
    legend({'no dummy','with dummy ($\phi^\ast$)'},'Location','best'); legend boxoff;

    subplot(2,3,2); hold on; box on;
    plot(hz, irf_nd(:,2),'--','Color',co(1,:),'LineWidth',lw);
    plot(hz, irf_wd(:,2),'-.','Color',co(2,:),'LineWidth',lw);
    yline(0,':',Color=[.5 .5 .5]); title('Monetary IRF: inflation $\pi$');
    xlabel('quarters'); ylabel('ann.\ \% dev.'); xlim([0 Hirf]); xticks(0:4:Hirf);

    subplot(2,3,3); hold on; box on;
    % one profile marginal-likelihood curve per pandemic period, each relative to
    % its own max (peak = that period's optimal phi_h). A sharp peak at small phi
    % indicates a period that is strongly downweighted; a flat curve, little.
    pc = [0.85 0.10 0.10; 0.10 0.55 0.20; 0.00 0.45 0.85];  % Q1,Q2,Q3
    qlab = {'2020Q1','2020Q2','2020Q3'};
    hL = gobjects(1,P.h);
    for hh = 1:P.h
        hL(hh) = semilogx(P.grid, P.llrel(hh,:),'-','Color',pc(hh,:),'LineWidth',lw);
        [~,ig] = max(P.llrel(hh,:));           % marker at each curve's own peak
        plot(P.grid(ig), P.llrel(hh,ig),'o','Color',pc(hh,:), ...
             'MarkerFaceColor',pc(hh,:),'MarkerSize',6);
    end
    set(gca,'XScale','log'); ylim([-15 1.5]); % zoom to peak; large-phi cliff off-frame
    title('Per-period profile marg.\ lik.\ ($\phi_h$)');
    xlabel('$\phi_h$ (period-$h$ precision)'); ylabel('log marg.\ lik.\ (rel.\ max)');
    legend(hL, qlab,'Location','best'); legend boxoff;

    subplot(2,3,[4 5]); hold on; box on;
    zi = dates>=datetime(2015,1,1);
    % normalize each model's smoothed monetary shock by its own pre-2020 sd, so
    % the y-axis reads "shock in std-dev units" and the two models are comparable
    % despite different sigma_m (structural priors keep sigma_m sensible).
    en = out_nd.eps_smooth(:,3) / std(out_nd.eps_smooth(dates<datetime(2020,1,1),3),'omitnan');
    ew = out_wd.eps_smooth(:,3) / std(out_wd.eps_smooth(dates<datetime(2020,1,1),3),'omitnan');
    plot(dates(zi), en(zi),'--','Color',co(1,:),'LineWidth',lw);
    plot(dates(zi), ew(zi),'-.','Color',co(2,:),'LineWidth',lw);
    yline(0,':',Color=[.5 .5 .5]);
    xline(dates(pand(1)),':',Color=[.4 .4 .4]); xline(dates(pand(end)),':',Color=[.4 .4 .4]);
    title('Smoothed monetary shock $\varepsilon_m$ (own pre-2020 s.d.\ units)');
    ylabel('std.\ dev.');
    legend({'no dummy','with dummy'},'Location','best'); legend boxoff;

    subplot(2,3,6); hold on; box on;
    dwedge = reshape(out_wd.d_smooth, 3, []).';   % h-by-3 pandemic wedge [x pi i]
    wcol = [0.204 0.298 0.492;    % x  : indigo
            0.831 0.522 0.157;    % pi : ochre
            0.400 0.560 0.360];   % i  : sage
    b = bar(1:3, dwedge, 'grouped', 'EdgeColor','none');
    for kk = 1:numel(b), b(kk).FaceColor = wcol(kk,:); end
    title('Estimated pandemic wedge $d$ (with dummy)');
    set(gca,'XTick',1:3,'XTickLabel',{'2020Q1','2020Q2','2020Q3'});
    legend(b, {'$x$','$\pi$','$i$'},'Location','best'); legend boxoff; ylabel('obs.\ units');

    outpng = fullfile('figures', ['usdata_' tag '.png']);
    outpdf = fullfile('figures', ['usdata_' tag '.pdf']);
    exportgraphics(fig, outpng, 'Resolution',150);
    exportgraphics(fig, outpdf, 'ContentType','vector');
    close(fig);
    save(fullfile('figures',['usdata_' tag '.mat']), ...
        'th_nd','th_wd','phi_star','phi_grid','P','irf_nd','irf_wd', ...
        'out_nd','out_wd','dates','pand','D','meas_sd');
end
end

% ---- summary table across combos ----
fprintf('\n\n================ US-DATA SUMMARY (per-period phi) ================\n');
fprintf('%-24s %7s %7s %7s %9s %9s %7s %7s\n', ...
        'combo','phi_Q1','phi_Q2','phi_Q3','IRFx_nd','IRFx_wd','spk_nd','spk_wd');
for r = 1:size(summary,1)
    fprintf('%-24s %7.3g %7.3g %7.3g %9.3f %9.3f %7.1f %7.1f\n', summary{r,:});
end
fprintf('spk = max|2020 monetary shock| / pre-2020 sd\n');

% ====================================================================== %
% Shared model/prior logic lives in standalone files (also used by
% run_dsge_posterior.m): nk_struct_par, nk_logprior, nk_logpost.
function [obj, out] = kalman_ll(th, y, opt)
% Objective maximized in theta = data (marginal) log-likelihood + structural
% DSGE log-prior. phi selection reads out.loglik (pure data marginal lik).
    [obj, ll, out] = nk_logpost(th, y, opt);
    if ~isfinite(obj), obj = -1e10; end
    if ~isfield(out,'loglik') || ~isfinite(ll), out.loglik = -1e10; end
end
% parameter transforms (unconstrained <-> natural)
function x = pack(th)
    x = [ log(th(1)), log(th(2)-1), logit(th(3)), logit(th(4)), logit(th(5)), ...
          log(th(6)), log(th(7)), log(th(8)) ];
end
function th = unpack(x)
    th = [ exp(x(1)), 1+exp(x(2)), ilogit(x(3)), ilogit(x(4)), ilogit(x(5)), ...
           exp(x(6)), exp(x(7)), exp(x(8)) ];
end
function z = logit(p),  z = log(p./(1-p)); end
function p = ilogit(z), p = 1./(1+exp(-z)); end
