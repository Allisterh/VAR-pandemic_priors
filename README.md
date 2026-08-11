# pandemic_priors

A simple, easy, and flexible way of estimating Bayesian VARs taking into consideration the pandemic period, as a Minnesota prior with time dummies, available in MATLAB, Julia, Python, and R.  Codes embed a test for the optimal level of shrinkage for the pandemic period.

Also available are a DSGE extension of the Pandemic Priors, and MATLAB extensions of the Giannone, Lenza, and Primiceri (2015) optimal priors and the Chan (2022) asymmetric conjugate priors.

Paper available at my website: www.danilocascaldigarcia.com

Use of code for research purposes is permitted as long as proper reference to source is given.

This version: August 2026

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Repository structure:

Each language folder is self-contained and has its own README with details and instructions.

MATLAB/    Main implementation.  Run run_pandemic_priors.m.  See MATLAB/README.md.

Julia/     Main implementation.  Run run_pandemic_priors.jl.  See Julia/README.md.

Python/    Main implementation.  Run run_pandemic_priors.py.  See Python/README.md.

R/         Main implementation.  Run run_pandemic_priors.R.  See R/README.md.

DSGE/      DSGE extension: the Pandemic Priors dummies in the measurement equation of a small New Keynesian model, on simulated and US data.  See DSGE/README.md.

Other_implementations/    MATLAB extensions of the Giannone, Lenza, and Primiceri (2015) optimal priors and the Chan (2022) asymmetric conjugate priors.

Legacy_files/    Earlier versions of the codes in MATLAB, Julia, Python, and R, kept for reference.  See Legacy_files/README.md.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The main programs perform the Pandemic Priors Bayesian VAR estimation with time dummies on the pandemic period, and identify an EBP shock with a recursive Cholesky structure, where EBP is ordered first.  All options are set in the "USER SETTINGS" block at the top of each run_pandemic_priors script; see the language folder's README for the full list.

"covid_periods" defines how many monthly dummies to include (starting from and including March 2020).  Set to zero to run a conventional Minnesota Prior as in Banbura, Giannone, and Reichlin (2010).

"phi" is the Gaussian prior precision on the pandemic dummy coefficients, and defines how much signal the econometrician would like to take from the pandemic period.  With phi close to zero the time dummies are "active," soaking all the pandemics variance; with phi close to infinity the time dummies are "inactive," and the model boils down to a conventional Minnesota Prior.

"phi_mode" controls how phi is chosen: "single" selects one phi on a grid by marginal likelihood, "perperiod" selects one phi per pandemic month (phi_1 to phi_h) with the optimizer, or a fixed scalar/vector can be supplied directly (a value close to zero leaves the dummies uninformative).  The Minnesota tightness (lambda), sum-of-coefficients tightness (tau), and co-persistence prior (coper) can likewise be fixed or selected by marginal likelihood.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
References:

Bańbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector auto regressions. Journal of applied Econometrics, 25(1), 71-92.

Chan, J. C. (2022). Asymmetric conjugate priors for large Bayesian VARs. Quantitative economics, 13(3), 1145-1169.

Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for vector autoregressions. Review of Economics and Statistics, 97(2), 436-451.
