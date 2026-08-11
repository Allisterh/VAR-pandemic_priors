% DIAG_ALL_SHOCKS  Smoothed structural shocks across the pandemic window.
%
%   Loads a saved US-data result and plots the smoothed demand (eps_a),
%   supply/cost-push (eps_s), and monetary (eps_m) structural shocks, with and
%   without the pandemic dummy, each in its own pre-2020 s.d. units. Shows which
%   shocks absorb the 2020 wedge in the no-dummy fit.
%
%   Usage:  >> diag_all_shocks            % default combination below
%           >> diag_all_shocks('GDPDEF_FEDFUNDS')

function diag_all_shocks(tag)
if nargin<1, tag = 'PCEPILFE_FEDFUNDS'; end
here = fileparts(mfilename('fullpath')); cd(here);
addpath(fullfile(here,'utils'));                        % helper functions
set(0,'defaulttextinterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');

S = load(fullfile('figures',['usdata_' tag '.mat']));
dates = S.dates; pand = S.pand;
en = S.out_nd.eps_smooth;  ew = S.out_wd.eps_smooth;   % T-by-3 [eps_a eps_s eps_m]
pre = dates < datetime(2020,1,1);
lab = {'Demand $\varepsilon_a$','Supply/cost-push $\varepsilon_s$','Monetary $\varepsilon_m$'};

% normalize each series by its own pre-2020 s.d.
nrm = @(v) v ./ std(v(pre),'omitnan');

fprintf('\n=== %s: smoothed shocks in the pandemic window (own pre-2020 s.d. units) ===\n', tag);
fprintf('%-22s %8s %8s %8s | %8s %8s %8s\n','quarter','a_nd','s_nd','m_nd','a_wd','s_wd','m_wd');
qn = {'2020Q1','2020Q2','2020Q3'};
EN = [nrm(en(:,1)) nrm(en(:,2)) nrm(en(:,3))];
EW = [nrm(ew(:,1)) nrm(ew(:,2)) nrm(ew(:,3))];
for k=1:3
    r = pand(k);
    fprintf('%-22s %8.2f %8.2f %8.2f | %8.2f %8.2f %8.2f\n', qn{k}, EN(r,:), EW(r,:));
end

fig = figure('Position',[80 80 1200 360],'Color','w','Visible','off');
co = [0.85 0.10 0.10; 0.00 0.45 0.85]; lw = 1.8;
zi = dates>=datetime(2015,1,1);
for j = 1:3
    subplot(1,3,j); hold on; box on;
    plot(dates(zi), EN(zi,j),'--','Color',co(1,:),'LineWidth',lw);
    plot(dates(zi), EW(zi,j),'-.','Color',co(2,:),'LineWidth',lw);
    yline(0,':',Color=[.5 .5 .5]);
    xline(dates(pand(1)),':',Color=[.4 .4 .4]); xline(dates(pand(end)),':',Color=[.4 .4 .4]);
    title(lab{j}); ylabel('std.\ dev.');
    if j==1, legend({'no dummy','with dummy'},'Location','best'); legend boxoff; end
end
out = fullfile('figures',['diag_shocks_' tag '.png']);
exportgraphics(fig, out, 'Resolution',150);
exportgraphics(fig, fullfile('figures',['diag_shocks_' tag '.pdf']), 'ContentType','vector');
close(fig);
fprintf('\nfigure saved: %s (+ .pdf)\n', out);
end
