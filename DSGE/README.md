# Pandemic Priors in a DSGE model — MATLAB code

Companion code for the DSGE extension of Cascaldi-Garcia, D., "Pandemic Priors."
It shows how the same pandemic-dummy idea used in the BVAR carries over to a
structural state-space model: the dummy coefficients enter the **measurement
equation** of a linearized three-equation New Keynesian model, where they act as
measurement-error shocks localized to the pandemic quarters, so the pandemic
observations no longer force the structural shocks to explain the 2020 collapse
and rebound.

The main implementation is **self-contained MATLAB** (no Dynare and no toolboxes
required): the model is solved in closed form by undetermined coefficients
(`nk_solve`), the pandemic dummies are integrated out analytically inside a
linear Gaussian Kalman filter (`nk_kalman`), and the shrinkage parameter phi is
selected by maximizing the resulting marginal likelihood, the state-space
analogue of the closed-form BVAR marginal likelihood. A **Dynare version of the
model** (`nk_pandemic.mod`) is also provided; it reproduces the same policy rule
and impulse responses, for users who prefer to work from a `.mod` file.

## Requirements

- MATLAB (R2019b+). No toolboxes required for the core; the drivers use
  `exportgraphics` for the figures.

## How to run

The scripts share results through the `figures/` folder, so run them in order.

**Simulated data (validation on a known truth):**
```matlab
run_dsge_simulated      % writes figures/simulated_results.mat and the figure
run_dsge_posterior_sim  % posterior figure; reads figures/simulated_results.mat
```

**US data (application):**
```matlab
run_dsge_usdata         % writes figures/usdata_<combo>.mat and per-combo figures
run_dsge_posterior      % posterior figure; reads figures/usdata_<combo>.mat
diag_all_shocks         % smoothed-shock diagnostic; reads figures/usdata_<combo>.mat
```

All output (figures and intermediate `.mat` files) is written to `figures/`.

## Files

```
run_dsge_simulated.m      simulated-data driver
run_dsge_posterior_sim.m  simulated-data posterior figure
run_dsge_usdata.m         US-data driver
run_dsge_posterior.m      US-data posterior figure
diag_all_shocks.m         smoothed structural-shock diagnostic

nk_pandemic.mod   Dynare version of the model (optional; reproduces the policy
                  rule and impulse responses of the self-contained MATLAB solver)

utils/            helper functions used by the drivers (added to the path
                  automatically):
    nk_params.m       baseline (quarterly, Gali 2015) calibration
    nk_struct_par.m   list / bounds of the estimated structural parameters
    nk_pack.m         pack a parameter struct into a vector
    nk_unpack.m       unpack a vector back into a parameter struct
    nk_solve.m        closed-form RE solution + linear state-space representation
    nk_irf.m          impulse responses from the state-space solution
    nk_kalman.m       Kalman filter/smoother with the pandemic dummies (phi) in
                      the measurement equation; returns the marginal log-lik
    nk_logprior.m     prior density over the structural parameters
    nk_logpost.m      log posterior (= log-likelihood + log-prior)
    nk_fit_phi.m      estimate (theta, phi_1..phi_h) by maximizing the marginal lik
    nk_profile_phi.m  profile marginal-likelihood curves in phi
    nk_mh.m           random-walk Metropolis sampler for the posterior figures
    nk_posterior_fig.m plotting helper for the posterior densities
    nk_dispname.m     display names for the variables (used in the figures)
    load_nk_data.m    load and demean the US data (output gap, inflation, rate)

data/             US data (FRED CSVs + Wu-Xia shadow rate) and build_data_nk.m
figures/          output folder (figures and intermediate .mat results)
```

## Data

The `data/` folder contains the FRED series used for the US-data application (real GDP and
potential GDP for the output gap, GDP deflator / core-CPI / core-PCE inflation,
the federal funds rate, and the Wu–Xia shadow rate), the assembled
`data_nk.csv`, and `build_data_nk.m` which reconstructs it. See
`data/FRED_DOWNLOAD_INSTRUCTIONS.txt` for the sources.

## Citation

Cascaldi-Garcia, D., "Pandemic Priors."
Use of the code for research purposes is permitted with proper reference.
