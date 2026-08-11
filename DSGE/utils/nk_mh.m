function chain = nk_mh(logpost, th0, step, ndraw, nburn, label)
% NK_MH  Random-walk Metropolis-Hastings on the NK structural parameters theta.
%
%   chain = NK_MH(logpost, th0, step, ndraw, nburn, label)
%
%   Proposals are made in the unconstrained space (via nk_pack/nk_unpack: log for
%   kappa/sig_*, log(phi_pi-1), logit for rho_*), so positivity and rho in (0,1)
%   hold automatically; the change-of-variables Jacobian is added so the target
%   is the posterior in natural coordinates. Step scale is adapted toward ~30%
%   acceptance during burn-in only. Shared by run_dsge_posterior[_sim].m.
%
%   INPUTS
%     logpost  function handle th (natural 8-vec) -> log-posterior (nk_logpost)
%     th0      1-by-8 start (natural units)
%     step     1-by-8 initial RW std devs in unconstrained space
%     ndraw    kept draws ; nburn burn-in draws
%     label    char, for the acceptance-rate printout
%
%   OUTPUT
%     chain    ndraw-by-8 posterior draws in natural units

npar = numel(th0);
x  = nk_pack(th0);
lp = logpost(th0) + logjac(x);
ii = 0;  while ~isfinite(lp) && ii<50            % nudge to a valid start
    x = x + 0.01*randn(1,npar); lp = logpost(nk_unpack(x)) + logjac(x); ii=ii+1;
end
s  = step;  acc = 0;  accwin = 0;
total = nburn + ndraw;
chain = zeros(ndraw, npar);
for it = 1:total
    xp  = x + s.*randn(1,npar);
    thp = nk_unpack(xp);
    lpp = logpost(thp) + logjac(xp);
    if log(rand) < (lpp - lp)
        x = xp; lp = lpp; acc=acc+1; accwin=accwin+1;
    end
    if it<=nburn && mod(it,250)==0               % adapt during burn-in only
        rate_win = accwin/250;  accwin=0;
        s = s * exp( (rate_win-0.30)*0.8 );
    end
    if it>nburn, chain(it-nburn,:) = nk_unpack(x); end
end
fprintf('  [%-10s] acc rate %.2f  (final step scale %.3f)\n', ...
        label, acc/total, mean(s));
end

% ---------------------------------------------------------------------- %
function lj = logjac(x)
% log|d theta / d x| for the nk_pack/nk_unpack transform.
%   kappa,phi_pi-1,sig_* : d(exp)/dx = exp(x)  -> +x
%   rho_* logit          : d ilogit/dx = p(1-p) -> log p + log(1-p)
lj = x(1) + x(2) + x(6) + x(7) + x(8);
for j = [3 4 5]
    p = 1./(1+exp(-x(j)));  lj = lj + log(p) + log(1-p);
end
end
