function sol = nk_solve(par)
% NK_SOLVE  Closed-form solution of the 3-equation New Keynesian model (gap form).
%
%   sol = NK_SOLVE(par) returns the rational-expectations solution and a linear
%   state-space representation of the model. Because the three driving processes
%   (r_nat, u_s, v_m) are AR(1) and the model is linear, the equilibrium is linear
%   in the states and solvable by undetermined coefficients:
%
%       x_t  = a_a*r_nat_t + a_s*u_s_t + a_m*v_m_t
%       pi_t = b_a*r_nat_t + b_s*u_s_t + b_m*v_m_t
%
%   The solution is exact (no perturbation) and inexpensive to call inside the
%   phi optimizer.
%
%   INPUT  par : struct with fields
%       beta sigma kappa phi_pi phi_x rho_a rho_s rho_m sig_a sig_s sig_m
%
%   OUTPUT sol : struct with fields
%       det      : logical, true if the model is determinate (Taylor principle)
%       a, b     : 1x3 policy-rule coefficient vectors, ordered [r_nat u_s v_m]
%       F        : 3x3 state transition,  s_t = F*s_{t-1} + eps_t
%       Q        : 3x3 structural shock covariance (diagonal)
%       Z        : 3x3 measurement matrix, [x;pi;i] = Z*s_t
%       obsnames : {'x','pi','i'}
%       statenames, shocknames : cellstr labels
%
%   The state vector is s = [r_nat; u_s; v_m] and the structural innovations are
%   eps = [eps_a; eps_s; eps_m] (so G = I: innovations enter the states directly,
%   which makes the smoothed states' innovations the structural shocks).
%
%   See also NK_IRF, NK_KALMAN.

% --- unpack ---------------------------------------------------------------
beta   = par.beta;   sigma = par.sigma;  kappa = par.kappa;
phi_pi = par.phi_pi; phi_x = par.phi_x;
rho    = [par.rho_a, par.rho_s, par.rho_m];
sig2   = [par.sig_a^2, par.sig_s^2, par.sig_m^2];

% --- determinacy (standard 3-eq NK condition) -----------------------------
% Determinate iff kappa*(phi_pi - 1) + (1 - beta)*phi_x > 0.
sol.det = ( kappa*(phi_pi - 1) + (1 - beta)*phi_x ) > 0;

% --- undetermined coefficients: one 2x2 solve per driver ------------------
% For a driver with persistence rho, [a;b] solve
%     [ (1-rho)+phi_x/sigma   (phi_pi-rho)/sigma ] [a]   [rhs1]
%     [ -kappa                 1-beta*rho        ] [b] = [rhs2]
% with (rhs1,rhs2) = (1/sigma,0) for r_nat, (0,1) for u_s, (-1/sigma,0) for v_m.
rhs = [ 1/sigma  0        -1/sigma ;    % row 1 (IS forcing)
        0        1         0        ];  % row 2 (NKPC forcing)

a = zeros(1,3);  b = zeros(1,3);
for k = 1:3
    M = [ (1-rho(k)) + phi_x/sigma ,  (phi_pi - rho(k))/sigma ;
          -kappa                   ,  1 - beta*rho(k)         ];
    ab = M \ rhs(:,k);
    a(k) = ab(1);
    b(k) = ab(2);
end
sol.a = a;  sol.b = b;

% --- state-space matrices -------------------------------------------------
sol.F = diag(rho);
sol.Q = diag(sig2);

% Measurement: [x; pi; i] as linear functions of s = [r_nat; u_s; v_m].
% i = phi_pi*pi + phi_x*x + v_m  ->  phi_pi*b + phi_x*a + [0 0 1].
i_row = phi_pi*b + phi_x*a + [0 0 1];
sol.Z = [ a ; b ; i_row ];

% --- labels ---------------------------------------------------------------
sol.obsnames   = {'x','pi','i'};
sol.statenames = {'r_nat','u_s','v_m'};
sol.shocknames = {'eps_a','eps_s','eps_m'};
end
