function [phi_use, lambda, tau, coper] = pp_select(phi_mode, lambda_mode, tau_mode, ...
        X, Y, Yraw, nAR, constant, delta, epsilon, covid_periods, phi_grid, coper_mode)
% PP_SELECT  Choose the Pandemic-Priors hyperparameters (phi, lambda, tau, coper)
%            by maximizing the fast marginal likelihood (pp_logml).
%
%   [phi_use,lambda,tau,coper] = pp_select(phi_mode,lambda_mode,tau_mode, ...
%                        X,Y,Yraw,nAR,constant,delta,epsilon,covid_periods,phi_grid,coper_mode)
%
%   phi_mode
%     'single'    : one phi, chosen on phi_grid
%     'perperiod' : phi_1..phi_h, chosen by the optimiser
%     <scalar/vec>: a fixed phi supplied directly (no phi search)
%
%   lambda_mode   : a scalar value (e.g. 0.2, the default), or 'optimal'
%   tau_mode      : a scalar value, '10lambda' (default; tau = 10*lambda), or 'optimal'
%   coper_mode    : a scalar co-persistence tightness (0 = off, the default), or 'optimal'
%
%   How the free hyperparameters are searched:
%     * phi_mode 'single' (or a fixed phi): GRID search over every free dimension,
%       using phi_grid for phi/lambda/tau/coper alike (as in the paper).
%     * phi_mode 'perperiod': the phi_h enter a bounded fminsearch, and any 'optimal'
%       lambda/tau/coper are appended as ADDITIONAL search parameters (base-MATLAB,
%       no Optimization Toolbox).
%   When lambda is 'optimal' and tau is '10lambda', tau is not a separate dimension:
%   it is reconstructed as 10*lambda at each candidate lambda (grid) or each function
%   evaluation (fminsearch).
%
%   All searches share the bound [min(phi_grid),max(phi_grid)] (= [1e-3,5] by
%   default); widen phi_grid if an optimum sits at the boundary. Note 'optimal'
%   coper therefore searches a positive tightness (it never selects coper = 0 /
%   off); leave coper_mode = 0 to keep the co-persistence prior switched off.
%
%   Danilo Cascaldi-Garcia

isstr  = @(s) ischar(s) || isstring(s);
lam_opt = isstr(lambda_mode) && strcmpi(lambda_mode,'optimal');
tau_opt = isstr(tau_mode)    && strcmpi(tau_mode,'optimal');
tau_tie = isstr(tau_mode)    && any(strcmpi(tau_mode,{'10lambda','10*lambda'}));
cop_opt = isstr(coper_mode)  && strcmpi(coper_mode,'optimal');

% Fixed values (used only when the corresponding mode is not 'optimal'/tied)
lambda_fix = 0; if ~lam_opt;            lambda_fix = lambda_mode; end
tau_fix    = 0; if ~tau_opt && ~tau_tie; tau_fix    = tau_mode;    end
coper_fix  = 0; if ~cop_opt;            coper_fix  = coper_mode;  end

lb = min(phi_grid); ub = max(phi_grid);           % shared search bounds
g    = @(z) lb + (ub-lb)./(1+exp(-z));            % logit^-1 : R -> (lb,ub)
ginv = @(p) log((p-lb)./(ub-p));

is_perperiod = isstr(phi_mode) && strcmpi(phi_mode,'perperiod');
is_single    = isstr(phi_mode) && strcmpi(phi_mode,'single');

if is_perperiod
    %====================== fminsearch over phi_1..phi_h (+ free lambda/tau) =====
    h = covid_periods;
    idx_lam = 0; idx_tau = 0; idx_cop = 0; npar = h;
    if lam_opt; npar = npar+1; idx_lam = npar; end
    if tau_opt; npar = npar+1; idx_tau = npar; end
    if cop_opt; npar = npar+1; idx_cop = npar; end

    % Objective (-logML) via a local function; all state passed explicitly.
    obj = @(z) negML_pp(z,g,h,idx_lam,idx_tau,idx_cop,lam_opt,tau_opt,tau_tie,cop_opt, ...
              lambda_fix,tau_fix,coper_fix,X,Y,Yraw,nAR,constant,delta,epsilon,covid_periods);

    opts = optimset('Display','off','MaxFunEvals',4000,'MaxIter',4000);
    best = inf; zbest = [];
    for s0 = [0.3 1.0 1.7]                          % multi-start (phi units)
        z0 = ginv(s0)*ones(1,npar);
        if idx_lam; z0(idx_lam) = ginv(0.2);       end   % lambda start ~ 0.2
        if idx_tau; z0(idx_tau) = ginv(min(2,ub)); end   % tau start
        if idx_cop; z0(idx_cop) = ginv(1);         end   % coper start ~ 1
        [zh,f] = fminsearch(obj,z0,opts);
        if f < best; best = f; zbest = zh; end
    end
    phi_use = g(zbest(1:h));
    if lam_opt; lambda = g(zbest(idx_lam)); else; lambda = lambda_fix; end
    if     tau_opt; tau = g(zbest(idx_tau));
    elseif tau_tie; tau = 10*lambda;
    else;           tau = tau_fix;
    end
    if cop_opt; coper = g(zbest(idx_cop)); else; coper = coper_fix; end
    clear obj                                       % release the ML handle

else
    %====================== grid search (single phi, or fixed phi) ===============
    % Warn about combinatorial cost: the grid path nests one loop per 'optimal'
    % dimension, so evaluations multiply. With 2+ extra optimal dimensions the
    % 'perperiod' path (bounded fminsearch, no grid blow-up) is far cheaper.
    n_extra_opt = lam_opt + tau_opt + cop_opt;
    if n_extra_opt >= 2
        nphi = numel(phi_grid); if ~is_single; nphi = 1; end
        nevals = nphi * (lam_opt*numel(phi_grid) + ~lam_opt) ...
                      * (tau_opt*numel(phi_grid) + ~tau_opt) ...
                      * (cop_opt*numel(phi_grid) + ~cop_opt);
        warning('pp_select:gridBlowup', ...
            ['Grid search with %d ''optimal'' hyperparameters (+phi) implies ~%d ' ...
             'marginal-likelihood evaluations. Consider phi_mode=''perperiod'', which ' ...
             'optimises lambda/tau/coper as continuous fminsearch dimensions instead ' ...
             'of a nested grid.'], n_extra_opt, nevals);
    end
    if is_single
        phi_cands = num2cell(phi_grid);             % search phi over the grid
    else
        phi_cands = {phi_mode};                     % fixed phi (scalar or vector)
    end
    if lam_opt; lam_vals = phi_grid; else; lam_vals = lambda_fix; end
    if cop_opt; cop_vals = phi_grid; else; cop_vals = coper_fix; end

    best = -inf; phi_use = phi_cands{1}; lambda = lam_vals(1); tau = NaN; coper = cop_vals(1);
    for il = 1:numel(lam_vals)
        lam = lam_vals(il);
        if     tau_tie; tau_vals = 10*lam;          % reconstructed at each lambda
        elseif tau_opt; tau_vals = phi_grid;
        else;           tau_vals = tau_fix;
        end
        for it = 1:numel(tau_vals)
            ta = tau_vals(it);
            for ic = 1:numel(cop_vals)
                cp = cop_vals(ic);
                for ip = 1:numel(phi_cands)
                    ml = pp_logml(X,Y,Yraw,nAR,constant,delta,lam,ta,epsilon,phi_cands{ip},covid_periods,cp);
                    if ml > best
                        best = ml; phi_use = phi_cands{ip}; lambda = lam; tau = ta; coper = cp;
                    end
                end
            end
        end
    end
end

%====================== report ==================================================
if isscalar(phi_use)
    phistr = sprintf('%.3f',phi_use);
else
    phistr = ['[' strjoin(compose('%.3f',phi_use(:)'),', ') ']'];
end
fprintf('Selected phi = %s | lambda = %.3f | tau = %.3f | coper = %.3f\n',phistr,lambda,tau,coper);
end

% ---------------------------------------------------------------------------
function f = negML_pp(z,g,h,idx_lam,idx_tau,idx_cop,lam_opt,tau_opt,tau_tie,cop_opt, ...
        lambda_fix,tau_fix,coper_fix,X,Y,Yraw,nAR,constant,delta,epsilon,covid_periods)
% Per-period objective: unpack the search vector z, honour the lambda/tau/coper
% modes, return -logML. idx_* are 0 when that parameter is not searched.
phi = g(z(1:h));
if lam_opt; lam = g(z(idx_lam)); else; lam = lambda_fix; end
if     tau_opt; ta = g(z(idx_tau));
elseif tau_tie; ta = 10*lam;                        % reconstructed each evaluation
else;           ta = tau_fix;
end
if cop_opt; cp = g(z(idx_cop)); else; cp = coper_fix; end
f = -pp_logml(X,Y,Yraw,nAR,constant,delta,lam,ta,epsilon,phi,covid_periods,cp);
end
