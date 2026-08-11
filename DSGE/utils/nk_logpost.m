function [lpost, loglik, out] = nk_logpost(th, y, opt)
% NK_LOGPOST  Log-posterior kernel in theta for the NK model.
%
%   [lpost, loglik, out] = NK_LOGPOST(th, y, opt)
%
%   lpost  = loglik + nk_logprior(th)
%   loglik = data (marginal) log-likelihood from nk_kalman, with the pandemic
%            dummy coefficients d integrated out when opt.use_dummy is true, so
%            it is the marginal likelihood in phi (analogue of the BVAR log-ML).
%   out    = full nk_kalman output (smoothed states/shocks/d).
%
%   Shared by run_dsge_usdata.m (mode-finding, phi selection uses loglik) and
%   run_dsge_posterior.m (MCMC targets lpost). Returns -Inf on indeterminate or
%   out-of-support draws so callers can reject cleanly.
    par = nk_struct_par(th);
    sol = nk_solve(par);
    if ~sol.det || any(~isfinite(th)) || any(th([1 6 7 8])<=0) ...
                || any(th([3 4 5])<=0) || any(th([3 4 5])>=1)
        lpost=-Inf; loglik=-Inf; out=struct('loglik',-Inf); return;
    end
    out = nk_kalman(y, sol, opt);
    loglik = out.loglik;
    if ~isfinite(loglik), lpost=-Inf; loglik=-Inf; return; end
    lpr = nk_logprior(th);
    if ~isfinite(lpr), lpost=-Inf; return; end
    lpost = loglik + lpr;
end
