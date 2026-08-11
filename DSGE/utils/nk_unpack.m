function th = nk_unpack(x)
% NK_UNPACK  Unconstrained space -> natural structural params. Inverse of nk_pack.
%   Returns th = [kappa phi_pi rho_a rho_s rho_m sig_a sig_s sig_m].
th = [ exp(x(1)), 1+exp(x(2)), ilogit(x(3)), ilogit(x(4)), ilogit(x(5)), ...
       exp(x(6)), exp(x(7)), exp(x(8)) ];
end
function p = ilogit(z), p = 1./(1+exp(-z)); end
