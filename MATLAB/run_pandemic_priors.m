%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  PANDEMIC PRIORS — public replication code
%  Cascaldi-Garcia, D., "Pandemic Priors".
%
%  Minimal, single-core (no parfor) implementation. Estimates a conjugate
%  Normal-Inverse-Wishart BVAR augmented with n*h pandemic dummy coefficients
%  whose Gaussian prior precision is phi, then:
%     (1) selects phi (single, per-period, or fixed),
%     (2) draws the posterior and forecasts, and
%     (3) identifies the EBP shock (recursive/Cholesky, EBP ordered first).
%
%  The heavy lifting is in two helpers: utils/pp_logml.m (fast Cholesky-based
%  marginal likelihood for the phi search) and utils/pp_draws.m (vectorized
%  single-core posterior draws + forecast + IRF).
%
%  Use of code for research purposes is permitted with proper reference.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
set(0,'defaulttextinterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');
clear; close all;
addpath([cd '/utils']);

%% ============================ USER SETTINGS ==============================
nAR           = 12;        % VAR lags
covid_periods = 6;         % pandemic dummy months, from Mar/2020 (h)
nimp          = 12;        % IRF / forecast horizon (months)
rps           = 10000;     % posterior draws
shocks        = 1;         % recursive (Cholesky) shocks to compute. Either a
                           % count (1 = first shock, EBP) or a 0/1 selector over
                           % the variables, e.g. [1 0 0 1 0 0 0 0] = shocks 1 & 4.
                           % Only selected shocks are computed (no extra cost).
constant      = 1;         % include intercept
delta         = 1;         % prior mean of own first lag (1 = levels/random walk,
                           % 0 = white noise / differences). Can be a SCALAR applied
                           % to all variables, OR a 1 x nvar row vector to set it
                           % per variable, e.g. [1 1 0 1 1 1 1 0] to center rates on
                           % white noise and the rest on a random walk. It enters
                           % element-wise in both the Minnesota block (diag(sigma.*delta))
                           % and the sum-of-coefficients block (diag(delta.*mu)).
epsilon       = 0.001;     % diffuse prior on the constant
seed          = 1;         % rng seed for reproducible bands ([] to skip)
screen        = true;     % stationarity screen: reject draws with an explosive
                           % companion root (max|eig|>=1.01).
                           % true  -> default
                           % false -> keep every draw; faster loop, but
                           %          retained explosive draws fatten the tails

phi_mode    = 'perperiod'; % 'single'  : one phi, chosen on a grid
                           % 'perperiod': phi_1..phi_h, chosen by optimiser
                           % <scalar>  : a fixed phi (e.g. 0.1; ~0 = uninformative)
lambda_mode = 0.2;         % Minnesota tightness: a scalar (0.2 default) or 'optimal'
tau_mode    = '10lambda';  % sum-of-coefficients tightness: a scalar, '10lambda'
                           % (default, tau=10*lambda), or 'optimal'. Set tau_mode=0
                           % to drop the sum-of-coefficients prior entirely.
coper_mode  = 0;           % co-persistence / dummy-initial-observation prior
                           % (Sims-Zha 1998): a single extra dummy built from the
                           % average of the initial nAR observations, favouring a
                           % shared stochastic trend. 0 = OFF (paper default);
                           % a positive scalar sets a fixed tightness (smaller =
                           % tighter); or 'optimal' to select it by marginal
                           % likelihood alongside phi (and any optimal lambda/tau).
% Selection: for phi_mode 'single' every free hyperparameter is grid-searched
% (same grid as phi); for 'perperiod' free lambda/tau/coper join the fminsearch.
% When lambda is 'optimal' and tau is '10lambda', tau tracks 10*lambda
% automatically. 'optimal' coper searches a positive tightness (never 0/off) --
% leave coper_mode=0 to keep the co-persistence prior switched off.

% Data
data_vintage_index = 20260524;
time_vec  = datetime(1975,1,1):calmonths(1):datetime(2025,12,1);
log_vector = [0 1 0 1 1 1 1 0];   % 1 -> variable entered as 100*log
Yname = {'EBP','S\&P 500','Shadow Rate','PCE','PCE Price Index',...
         'Employment','Ind. Production','Unemp. Rate'};

%% ============================ BUILD Y, X ================================
data = readmatrix(['./data/Data_' num2str(data_vintage_index) '.xlsx']);
data = data(1:length(time_vec),2:end);

Yraw = data;
for ee=1:size(log_vector,2)
    if log_vector(ee)==1; Yraw(:,ee) = log(Yraw(:,ee))*100; end
end
[Traw,nvar] = size(Yraw);
Ylag = mlag2(Yraw,nAR);
if constant==1
    X = [Ylag(nAR+1:Traw,:) ones(Traw-nAR,1)];
else
    X = Ylag(nAR+1:Traw,:);
end
Y = Yraw(nAR+1:Traw,:);

% Pandemic dummy columns: identity block at the pandemic rows
covid_ind = find(time_vec==datetime(2020,3,1)) - nAR;
X = [X, zeros(size(X,1),covid_periods)];
X(covid_ind:covid_ind+covid_periods-1,end-covid_periods+1:end) = eye(covid_periods);

%% ============ SELECT HYPERPARAMETERS (phi, lambda, tau, coper) ==========
% The same 18-point grid is used for phi and for any grid-searched
% lambda/tau/coper.
phi_grid = [0.001 0.01 0.025 0.05 0.075 0.10 0.15 0.20 0.25 0.30 ...
            0.35 0.40 0.45 0.50 0.75 1 2 5];

tstart = tic;
[phi_use, lambda, tau, coper] = pp_select(phi_mode, lambda_mode, tau_mode, ...
        X, Y, Yraw, nAR, constant, delta, epsilon, covid_periods, phi_grid, coper_mode);
t_phi = toc(tstart);

%% ============================ ESTIMATE =================================
[~,~,xd,yd] = pandemicpriors(X,Y,Yraw,nAR,constant,delta,lambda,tau,epsilon,phi_use,covid_periods,coper);

tstart = tic;
S = pp_draws(X,Y,xd,yd,nAR,covid_periods,nimp,shocks,rps,seed,screen);
t_draw = toc(tstart);

screentxt = {'OFF','ON'};
fprintf('phi-search %.2fs | draws+fcst+IRF %.2fs | stationarity screen %s (discarded %d)\n',...
        t_phi,t_draw,screentxt{screen+1},S.discarded);

%% ============================ FIGURES ==================================
% Figure style: no super-title, legend boxoff + Location best, explicit palette.
grey   = [0.70 0.70 0.70];  greyE = [0.65 0.65 0.65];
red    = [0.85 0.11 0.11];  black = [0 0 0];
bands  = [50 16 84];
p_lines = floor(nvar/3); p_cols = ceil(nvar/p_lines);
xax = (1:nimp)';

% ---- Shock IRFs (one figure per requested shock; median + 68% band) ----
for r = S.shocks
    fig = figure('Color','w','Units','normalized','OuterPosition',[0 0 0.8 0.8],...
           'Name',[Yname{r} '-shock']);
    for uu=1:nvar
        subplot(p_lines,p_cols,uu)
        q = squeeze(pp_prctile(squeeze(S.irf(r,:,:,uu)),bands,1))';
        patch([xax;flipud(xax)],[q(1:nimp,2);flipud(q(1:nimp,3))],grey,'EdgeColor',greyE); hold on
        plot(xax,q(1:nimp,1),'-','Color',red,'LineWidth',3);
        plot(xax,zeros(nimp,1),':','Color',black);
        title(Yname{uu},'Interpreter','latex'); xlabel('months'); axis tight; hold off
        if uu==1; legend({'68\% band','median'},'Location','best','Interpreter','latex'); legend boxoff; end
    end
    exportgraphics(fig,sprintf('irf_shock%d.pdf',r),'ContentType','vector');
end

% ---- Forecast fan charts (median + 68% band) ----
fig = figure('Color','w','Units','normalized','OuterPosition',[0 0 0.8 0.8],'Name','forecast');
for uu=1:nvar
    subplot(p_lines,p_cols,uu)
    q = squeeze(pp_prctile(squeeze(S.fcst(:,uu,:)),bands,1))';
    patch([xax;flipud(xax)],[q(1:nimp,2);flipud(q(1:nimp,3))],grey,'EdgeColor',greyE); hold on
    plot(xax,q(1:nimp,1),'-','Color',red,'LineWidth',3);
    title(Yname{uu},'Interpreter','latex'); xlabel('months'); axis tight; hold off
    if uu==1; legend({'68\% band','median'},'Location','best','Interpreter','latex'); legend boxoff; end
end
exportgraphics(fig,'forecast.pdf','ContentType','vector');
fprintf('wrote irf_shock*.pdf and forecast.pdf\n');

fprintf('Done in %.2f seconds. A_post %dx%d, IRF %dx%dx%dx%d, forecast %dx%dx%d\n',...
        t_phi+t_draw,...
        size(S.A_post,1),size(S.A_post,2),...
        size(S.irf,1),size(S.irf,2),size(S.irf,3),size(S.irf,4),...
        size(S.fcst,1),size(S.fcst,2),size(S.fcst,3));
