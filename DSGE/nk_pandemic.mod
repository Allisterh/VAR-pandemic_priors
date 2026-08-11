// ===========================================================================
//  nk_pandemic.mod  -- 3-equation New Keynesian model (gap form), Dynare version
//
//  Dynare implementation of the model used in the DSGE application of the
//  Pandemic Priors. It fixes the model and calibration; running it produces the
//  policy rule (oo_.dr) and the monetary impulse responses (oo_.irfs). This is
//  the Dynare counterpart of the self-contained MATLAB solver (nk_solve.m), which
//  reproduces the same objects; the pandemic-dummy estimation itself is done in
//  the MATLAB run_dsge_*.m scripts.
//
//  Dynare 7.1. Log-deviations from steady state => steady state is zero
//  (analytical, no numerical steady-state solver needed).
//
//  Cascaldi-Garcia, D., "Pandemic Priors."
//  Use of code for research purposes is permitted with proper reference.
//  Danilo Cascaldi-Garcia.
// ===========================================================================

var
    x        // output gap
    pi       // inflation
    i        // nominal interest rate
    r_nat    // natural rate process (demand/technology driver)
    u_s      // cost-push process
    v_m      // monetary policy process
;

varexo
    eps_a    // supply/technology innovation (enters via natural rate)
    eps_s    // cost-push innovation
    eps_m    // monetary policy innovation
;

parameters
    beta sigma kappa phi_pi phi_x
    rho_a rho_s rho_m
    sig_a sig_s sig_m
;

// --- Calibration (quarterly, standard textbook Gali values)
beta   = 0.99;
sigma  = 1;
kappa  = 0.1275;      // = (1-theta)(1-beta*theta)/theta*(sigma+phi)
phi_pi = 1.5;
phi_x  = 0.125/4;     // = 0.03125
rho_a  = 0.5;
rho_s  = 0.5;
rho_m  = 0.5;

// --- Structural shock standard deviations (illustration scale)
sig_a  = 1.0;
sig_s  = 0.5;
sig_m  = 0.25;

model(linear);
    // (IS)        dynamic IS / Euler equation
    x = x(+1) - (1/sigma)*( i - pi(+1) - r_nat );

    // (NKPC)      New Keynesian Phillips curve
    pi = beta*pi(+1) + kappa*x + u_s;

    // (Taylor)    monetary policy rule
    i = phi_pi*pi + phi_x*x + v_m;

    // (nat rate)  demand/technology driver
    r_nat = rho_a*r_nat(-1) + eps_a;

    // (cost push) supply driver
    u_s = rho_s*u_s(-1) + eps_s;

    // (mon)       monetary policy driver
    v_m = rho_m*v_m(-1) + eps_m;
end;

// Steady state is identically zero (all variables are gaps / deviations).
initval;
    x     = 0;
    pi    = 0;
    i     = 0;
    r_nat = 0;
    u_s   = 0;
    v_m   = 0;
end;

steady;
check;

shocks;
    var eps_a; stderr sig_a;
    var eps_s; stderr sig_s;
    var eps_m; stderr sig_m;
end;

// First-order solution + IRFs. irf=20 quarters.
// order=1 (linear); nograph suppresses Dynare's plots (figures drawn in MATLAB).
stoch_simul(order=1, irf=20, nograph, noprint);
