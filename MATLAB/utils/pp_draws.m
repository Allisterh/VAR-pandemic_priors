function out = pp_draws(X,Y,xd,yd,nAR,covid_periods,nimp,shocks,rps,seed,screen)
% PP_DRAWS  Single-core posterior draws + forecast + Cholesky IRF for the
%           Pandemic-Priors conjugate BVAR (no parfor).
%
%   Single-core (no parfor) implementation of the posterior draws, forecasts,
%   and impulse responses for the conjugate Normal-Inverse-Wishart BVAR.
%
%   Speed technique: reuse one Cholesky factor of the
%   augmented cross-product XXst for the posterior mean and for the matrix-normal
%   coefficient draws (triangular solve, no inv), and reuse the inverse-Cholesky
%   of the IW scale across all draws. On a single core a tight for-loop with only
%   an inverse-Wishart draw + two chol + the companion stationarity eig per draw
%   is fast. Toolbox-free: uses pp_iwishrnd (no Statistics/ML Toolbox).
%
%   INPUTS
%     X,Y            data regressors / LHS (T-by-k, T-by-n) as built in the driver
%     xd,yd          dummy-observation matrices from pandemicpriors.m
%     nAR            VAR lags
%     covid_periods  number of pandemic dummy columns (h)
%     nimp           IRF / forecast horizon
%     shocks         which recursive (Cholesky) shocks to compute. Either a count
%                    (scalar k -> the first k shocks, 1..k) OR a 0/1 selector over
%                    the n variables, e.g. [1 0 0 1 0 0 0 0] = shocks 1 and 4 only.
%                    Only the selected shock columns are propagated, so unselected
%                    shocks add no cost. .irf keeps the full n-shock first
%                    dimension; rows for shocks not requested are left at zero.
%     rps            number of posterior draws
%     seed           (optional) rng seed for reproducible bands
%     screen         (optional, default true) explosive-root rejection sampler.
%                    true  -> reject draws with companion max|eig|>=1.01 (default).
%                             The eig of the companion matrix dominates the
%                             per-draw wall clock.
%                    false -> keep every draw (no eig). Much faster loop, but the
%                             retained explosive draws fatten the IRF/forecast tails.
%
%   OUTPUT (struct)
%     .A_post   posterior mean coefficients (k-by-n)      [deterministic]
%     .SSE_post posterior IW scale (n-by-n)               [deterministic]
%     .v1       posterior IW degrees of freedom           [deterministic]
%     .irf      (n-by-rps-by-nimp-by-n) sign-normalized IRFs; only the requested
%               shock rows are populated (others are zero)
%     .shocks   the resolved list of computed shock indices
%     .fcst     (rps-by-n-by-nimp) posterior predictive draws
%     .C_auto   (rps-by-n-by-n) first-own-lag coefficient block
%     .C_dummies(rps-by-(1+h)-by-n) constant + pandemic dummy coefficients
%     .discarded number of draws rejected by the stationarity screen
%
%   Danilo Cascaldi-Garcia

if nargin>=10 && ~isempty(seed); rng(seed); end
if nargin<11 || isempty(screen); screen = true; end   % default: reject explosive draws

[T,k] = size(X);          %#ok<ASGLU>  % k = n*nAR + covid_periods + 1
n     = size(Y,2);
nvar  = n;
sizeCF = nAR*nvar;        % companion index for the stationarity screen

% Resolve the shock selection into a list of column indices.
%   scalar k        -> first k shocks (1..k), backward-compatible
%   0/1 selector    -> the flagged variables, e.g. [1 0 0 1 ...] -> [1 4]
if isscalar(shocks)
    shock_idx = 1:shocks;
else
    shock_idx = find(shocks(:)' ~= 0);
end

Xst = [X; xd];
Yst = [Y; yd];

%============================ POSTERIORS (deterministic) ===================
XXst = Xst'*Xst;                 % = xd'*xd + X'*X
C    = chol(XXst,'lower');       % XXst = C*C'  (reused for mean and draws)
XYst = Xst'*Yst;                 % = xd'*yd + X'*Y
A_post   = C'\(C\XYst);          % = XXst \ XYst   (no inv)
RESID    = Yst - Xst*A_post;
SSE_post = RESID'*RESID;
SSE_post = (SSE_post+SSE_post')/2;              % numerical symmetry guard
v1       = size(Xst,1) + 2 - size(Xst,2);       % posterior IW degrees of freedom

% Prime the (toolbox-free) inverse-Wishart sampler once: chol of the scale
% matrix is reused across all draws (pp_iwishrnd replaces iwishrnd).
CSSE = chol(SSE_post,'lower');

%============================ DRAWS (single core) ==========================
irf      = zeros(n,rps,nimp,n);       % full n-shock first dim; only shock_idx filled
fcst     = zeros(rps,n,nimp);
C_auto   = zeros(rps,n,n);
C_dummies= zeros(rps,1+covid_periods,n);
discarded= 0;

% Forecast seed row: last observation's lags + constant/pandemic columns
Ylast = Y(end,:);
Xtail = [X(end,1:nvar*(nAR-1)), X(end,nvar*nAR+1:end)];  % lags 1..p-1 + const/covid cols

% Preallocate the companion once; only its top nvar rows change per draw.
A_companion_dr = zeros(nvar*nAR,nvar*nAR);
A_companion_dr(nvar+1:nvar*nAR,1:nvar*nAR-nvar) = eye(nvar*nAR-nvar);

for iii=1:rps
    % ---- draw (Sigma, coefficients), optional stationarity rejection ----
    control2 = 0;
    while control2==0
        sigmarep = pp_iwishrnd(SSE_post,v1,CSSE);       % draw Sigma (no toolbox)
        nbeta_dr = A_post + (C'\randn(k,n))*chol(sigmarep);   % matrix-normal
        A_companion_dr(1:nvar,:) = nbeta_dr(1:nvar*nAR,:)';
        if ~screen
            control2 = 1;                               % keep every draw (no eig)
        else
            ROOTS_a = sort(abs(eig(A_companion_dr)));
            if ROOTS_a(sizeCF) < 1.01
                control2 = 1;
            else
                discarded = discarded + 1;
            end
        end
    end
    A0hat = chol(sigmarep)';                            % Cholesky, EBP first

    C_auto(iii,:,:)    = nbeta_dr(1:nvar,1:nvar);
    C_dummies(iii,:,:) = nbeta_dr(nvar*nAR+1:end,:);

    % ---- forecast: iterate Y_for = X_for*nbeta_dr ----
    Y_for = Ylast;
    X_for = [Ylast, Xtail];
    for ww=1:nimp
        if ww>1
            Y_for = X_for*nbeta_dr;
            X_for = [Y_for, X_for(1:nvar*(nAR-1)), X_for(nvar*nAR+1:end)];
        end
        fcst(iii,:,ww) = Y_for;
    end

    % ---- IRFs: propagate the companion state (no repeated matrix powers) ----
    % imp(k,:) = first nvar rows of A_companion^(k-1) * U1(:,r); propagate the
    % vector z_k = A_companion * z_{k-1} instead of recomputing each power.
    % Only the requested shock columns are propagated -> unselected shocks free.
    for r = shock_idx
        imp = zeros(nimp,nvar);
        z = [A0hat(:,r); zeros(nvar*nAR-nvar,1)];      % impact of shock r only
        imp(1,:) = z(1:nvar)';
        for kk=2:nimp
            z = A_companion_dr*z;
            imp(kk,:) = z(1:nvar)';
        end
        if imp(1,r) < 0; imp = -imp; end               % own-impact positive
        irf(r,iii,:,:) = reshape(imp,[1 1 nimp nvar]);
    end
end

out = struct('A_post',A_post,'SSE_post',SSE_post,'v1',v1, ...
             'irf',irf,'fcst',fcst,'C_auto',C_auto,'C_dummies',C_dummies, ...
             'discarded',discarded,'shocks',shock_idx);
end
