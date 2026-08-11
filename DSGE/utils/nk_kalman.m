function out = nk_kalman(y, sol, opt)
% NK_KALMAN  Kalman filter/smoother for the NK state space, with pandemic
%            dummy coefficients integrated out (marginal likelihood in phi).
%
%   out = NK_KALMAN(y, sol, opt)
%
%   State space (states s = [r_nat; u_s; v_m], obs = [x; pi; i]):
%       s_t = F s_{t-1} + eps_t,     eps_t ~ N(0, Q)     (G = I: innovations ARE
%       y_t = Z s_t + M_t d + w_t,   w_t  ~ N(0, H)       the structural shocks)
%
%   The pandemic dummy coefficients d (n per pandemic date; nd = n*h total) enter
%   the measurement equation, active only at pandemic dates via the selector M_t.
%   This is the BVAR dummy scheme applied to the state-space model.
%
%   As in the BVAR, d has a Gaussian prior  d ~ N(0, diag(phi^{-1})). The state is
%   augmented with d (constant, prior var phi^{-1}) and the filter is run, so the
%   filter's log-likelihood equals the marginal likelihood p(y|theta,phi) with d
%   integrated out, the direct analogue of the BVAR log-ML. Maximising out.loglik
%   over phi gives the overall / per-period selection.  phi->0: diffuse d,
%   pandemic obs dropped (Schorfheide-Song); phi->inf: d pinned to 0, ordinary
%   estimation.
%
%   INPUT
%     y   : T-by-n observations (no missing values)
%     sol : output of nk_solve (uses sol.F, sol.Q, sol.Z)
%     opt : struct with fields
%             use_dummy   : logical, augment with pandemic dummies (default false)
%             pandemic    : 1-by-h indices of pandemic periods in 1..T (rows of y)
%             phi         : scalar (overall) OR 1-by-h vector (per-period) prior
%                           precision on the dummy coefficients
%             meas_sd     : measurement-error s.d. (scalar), default 1e-4 for
%                           numerical stability
%
%   OUTPUT
%     out.loglik    : (marginal) log-likelihood p(y|theta[,phi])
%     out.eps_smooth: T-by-n smoothed structural shocks (row t = eps_t; row 1 NaN)
%     out.s_smooth  : T-by-n smoothed states [r_nat u_s v_m]
%     out.d_smooth  : nd-by-1 smoothed dummy coefficients (E[d|y]); [] if no dummy
%     out.d_prior_sd: nd-by-1 prior s.d. of d (phi^{-1/2}); [] if no dummy
%
%   See also NK_SOLVE, NK_IRF.

[T, n] = size(y);
F = sol.F;  Q = sol.Q;  Z = sol.Z;
ns = size(F,1);                                   % # structural states (3)

if nargin < 3, opt = struct; end
use_dummy = isfield(opt,'use_dummy') && opt.use_dummy;
meas_sd   = 1e-4;  if isfield(opt,'meas_sd') && ~isempty(opt.meas_sd), meas_sd = opt.meas_sd; end
% meas_sd may be scalar (common) or 1-by-n / n-by-1 (per-observable) s.d.
if isscalar(meas_sd)
    H = (meas_sd^2) * eye(n);
else
    H = diag(meas_sd(:).^2);
end

% --- dummy-coefficient bookkeeping ---------------------------------------
if use_dummy
    pand = opt.pandemic(:).';        % 1-by-h period indices
    h    = numel(pand);
    nd   = n*h;                      % one dummy coefficient per (obs, pandemic date)
    % prior variance for each coefficient (overall phi or per-period phi_h)
    if isscalar(opt.phi)
        vd = repmat(1/opt.phi, nd, 1);
    else
        assert(numel(opt.phi)==h, 'phi must be scalar or length h.');
        vd = reshape(repmat(1./opt.phi(:).', n, 1), nd, 1);   % n coeffs share phi_p
    end
    % per-period measurement selectors M_t (n x nd), nonzero only at pandemic dates
    Mcell = cell(T,1);  [Mcell{:}] = deal(zeros(n,nd));
    for p = 1:h
        cols = (p-1)*n + (1:n);
        Mcell{pand(p)} = [zeros(n,(p-1)*n), eye(n), zeros(n,nd-p*n)];
    end
else
    nd = 0;  vd = [];  Mcell = [];
end

na = ns + nd;                                     % augmented state dimension

% --- augmented system matrices -------------------------------------------
Ta = blkdiag(F, eye(nd));                         % d is constant
Qa = blkdiag(Q, zeros(nd));                       % no innovation to d

% stationary covariance of the AR(1) states (diagonal F, Q)
P0s = diag( diag(Q) ./ (1 - diag(F).^2) );
P_pred = blkdiag(P0s, diag(vd));                  % t=1 prior: states stationary, d~N(0,vd)
a_pred = zeros(na,1);

% storage for smoother
a_filt = zeros(na,T);  P_filt = zeros(na,na,T);
a_prd  = zeros(na,T);  P_prd  = zeros(na,na,T);

loglik = 0;
log2pi = log(2*pi);
for t = 1:T
    a_prd(:,t) = a_pred;  P_prd(:,:,t) = P_pred;

    if use_dummy, C = [Z, Mcell{t}]; else, C = Z; end
    v  = y(t,:).' - C*a_pred;                     % innovation
    S  = C*P_pred*C' + H;                          % innovation variance
    S  = (S+S')/2;
    [Ls, pchol] = chol(S,'lower');
    if pchol ~= 0                                  % S not PD (optimizer strayed)
        S  = S + (1e-8 + 1e-6*trace(S)/n)*eye(n);  % symmetric jitter, then retry
        [Ls, pchol] = chol(S,'lower');
        if pchol ~= 0                              % still bad: abort this eval
            out = struct('loglik',-Inf,'s_smooth',[],'eps_smooth',[], ...
                         'd_smooth',[],'d_prior_sd',[]);
            return;
        end
    end
    Kt = (P_pred*C') / S;                          % Kalman gain

    a_filt(:,t)   = a_pred + Kt*v;
    P_filt(:,:,t) = P_pred - Kt*C*P_pred;
    P_filt(:,:,t) = (P_filt(:,:,t)+P_filt(:,:,t)')/2;

    % log-likelihood contribution
    u = Ls\v;
    loglik = loglik - 0.5*( n*log2pi + 2*sum(log(diag(Ls))) + (u.'*u) );

    % predict
    a_pred = Ta*a_filt(:,t);
    P_pred = Ta*P_filt(:,:,t)*Ta' + Qa;
    P_pred = (P_pred+P_pred')/2;
end

% --- RTS smoother ---------------------------------------------------------
a_sm = zeros(na,T);  P_sm = zeros(na,na,T);
a_sm(:,T) = a_filt(:,T);  P_sm(:,:,T) = P_filt(:,:,T);
for t = T-1:-1:1
    Ppred_next = P_prd(:,:,t+1);
    Jt = P_filt(:,:,t)*Ta' / Ppred_next;
    a_sm(:,t)   = a_filt(:,t)   + Jt*(a_sm(:,t+1)   - a_prd(:,t+1));
    P_sm(:,:,t) = P_filt(:,:,t) + Jt*(P_sm(:,:,t+1) - Ppred_next)*Jt';
end

% --- collect outputs ------------------------------------------------------
s_sm = a_sm(1:ns,:).';                             % T-by-ns smoothed states
% smoothed structural shocks: eps_t = s_t - F s_{t-1} (G=I); row 1 undefined
eps_sm = nan(T, ns);
for t = 2:T
    eps_sm(t,:) = ( s_sm(t,:).' - F*s_sm(t-1,:).' ).';
end

out.loglik     = loglik;
out.s_smooth   = s_sm;
out.eps_smooth = eps_sm;
if use_dummy
    out.d_smooth   = a_sm(ns+1:end, 1);            % E[d|y] (identical across t)
    out.d_prior_sd = sqrt(vd);
else
    out.d_smooth   = [];
    out.d_prior_sd = [];
end
end
