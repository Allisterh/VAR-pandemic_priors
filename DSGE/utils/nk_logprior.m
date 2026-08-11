function lp = nk_logprior(th)
% NK_LOGPRIOR  Structural (DSGE) log-prior on theta = [kappa phi_pi rho_a rho_s
%   rho_m sig_a sig_s sig_m]. Standard weakly-informative Bayesian-DSGE priors.
%
%   This is separate from the Pandemic Prior. The Pandemic Prior is the Gaussian
%   prior of precision phi on the dummy coefficients d (in the measurement
%   equation), integrated out inside nk_kalman. This log-prior sits on a disjoint
%   parameter block (theta) and only regularizes the structural parameters, the
%   DSGE counterpart of the standard Minnesota prior on the VAR coefficients.
%
%     kappa  ~ Gamma(mean 0.15, sd 0.10)        NKPC slope (flat)
%     phi_pi ~ Gamma(mean 1.50, sd 0.25)        Taylor inflation response
%     rho_*  ~ Beta (mean 0.50, sd 0.20)        shock persistences
%     sig_*  ~ log-normal(median 1, sigma 0.7)  shock std. devs. (weak)
lp = lpgamma(th(1),0.15,0.10) ...
   + lpgamma(th(2),1.50,0.25) ...
   + lpbeta (th(3),0.50,0.20) ...
   + lpbeta (th(4),0.50,0.20) ...
   + lpbeta (th(5),0.50,0.20) ...
   + lplogn (th(6),1.00,0.70) ...
   + lplogn (th(7),1.00,0.70) ...
   + lplogn (th(8),1.00,0.70);
end

function v = lpgamma(x,m,s)     % Gamma parameterized by (mean, sd)
    if x<=0, v=-1e10; return; end
    a = (m/s)^2;  b = s^2/m;
    v = (a-1)*log(x) - x/b - a*log(b) - gammaln(a);
end
function v = lpbeta(x,m,s)      % Beta parameterized by (mean, sd)
    if x<=0 || x>=1, v=-1e10; return; end
    k = m*(1-m)/s^2 - 1;  a = m*k;  b = (1-m)*k;
    v = (a-1)*log(x) + (b-1)*log(1-x) - betaln(a,b);
end
function v = lplogn(x,med,sig)  % log-normal parameterized by (median, sigma)
    if x<=0, v=-1e10; return; end
    v = -log(x*sig*sqrt(2*pi)) - (log(x)-log(med))^2/(2*sig^2);
end
