function x = nk_pack(th)
% NK_PACK  Natural structural params -> unconstrained space (for optimisers/MCMC).
%   th = [kappa phi_pi rho_a rho_s rho_m sig_a sig_s sig_m]
%   kappa,sig_* > 0 (log); phi_pi > 1 (log of phi_pi-1); rho_* in (0,1) (logit).
%   Inverse of nk_unpack. Shared by run_dsge_simulated/usdata/posterior + nk_fit_phi.
x = [ log(th(1)), log(th(2)-1), logit(th(3)), logit(th(4)), logit(th(5)), ...
      log(th(6)), log(th(7)), log(th(8)) ];
end
function z = logit(p), z = log(p./(1-p)); end
