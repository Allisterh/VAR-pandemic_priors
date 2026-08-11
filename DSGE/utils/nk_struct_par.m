function par = nk_struct_par(th)
% NK_STRUCT_PAR  Map the 8-vector of estimated params to the full NK par struct.
%   th = [kappa phi_pi rho_a rho_s rho_m sig_a sig_s sig_m]
%   Fixed (not estimated): beta, sigma, phi_x.  Shared by the US-data driver
%   (run_dsge_usdata.m) and the MCMC sampler (run_dsge_posterior.m).
par.beta=0.99; par.sigma=1; par.phi_x=0.125/4;
par.kappa=th(1); par.phi_pi=th(2);
par.rho_a=th(3); par.rho_s=th(4); par.rho_m=th(5);
par.sig_a=th(6); par.sig_s=th(7); par.sig_m=th(8);
end
