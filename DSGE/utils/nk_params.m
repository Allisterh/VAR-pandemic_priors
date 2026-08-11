function par = nk_params()
% NK_PARAMS  Baseline calibration of the 3-equation NK model.
%
%   Quarterly, standard textbook Gali (2015) values.

par.beta   = 0.99;
par.sigma  = 1;
par.kappa  = 0.1275;      % = (1-theta)(1-beta*theta)/theta*(sigma+phi)
par.phi_pi = 1.5;
par.phi_x  = 0.125/4;     % = 0.03125

par.rho_a  = 0.5;
par.rho_s  = 0.5;
par.rho_m  = 0.5;

par.sig_a  = 1.0;
par.sig_s  = 0.5;
par.sig_m  = 0.25;
end
