# Pandemic Priors — R implementation

R port of the Pandemic Priors replication code, from Cascaldi-Garcia, D.,
"Pandemic Priors." A one-to-one translation of the MATLAB version, using base R
for the core (only `readxl` is needed to read the example spreadsheet).

## Requirements

- R 4.x
- `readxl` (to read `data/Data_20260524.xlsx`)

```r
install.packages("readxl")
```

## How to run

```
Rscript run_pandemic_priors.R
```

Options are in the `USER SETTINGS` block at the top of `run_pandemic_priors.R`.

## Files

```
pandemic_priors.R        core functions (sourced by the driver)
run_pandemic_priors.R    driver: build the BVAR, select phi, draw, plot
data/Data_20260524.xlsx  example dataset (Jan 1975 – Dec 2025, 8 monthly series)
```

`pandemic_priors.R` provides `mlag2`, `pandemicpriors`, `pp_logml`, `pp_select`,
`pp_draws`, `pp_iwishrnd`, and `pp_prctile`, mirroring the MATLAB functions.

## Settings (in `run_pandemic_priors.R`)

All options live in the `USER SETTINGS` block, with the same inline notes.

Model / sampling:

- `nAR` — VAR lags; `covid_periods` — number of pandemic dummy months `h`;
  `nimp` — IRF/forecast horizon; `rps` — posterior draws; `constant` — include intercept.
- `shocks` — which recursive (Cholesky) shocks to compute: a count (`1` = first
  shock, EBP) or a 0/1 selector over the variables (e.g. `c(1,0,0,1,0,0,0,0)` =
  shocks 1 & 4). Only selected shocks are computed.
- `epsilon` — diffuse prior on the constant.
- `seed` — rng seed for reproducible bands (`NULL` to skip).
- `screen` — stationarity screen: `TRUE` (default) rejects draws with an explosive
  companion root (`max|eig| >= 1.01`); `FALSE` keeps every draw (faster, but the
  retained explosive draws fatten the tails).

Prior / shrinkage:

- `delta` — prior mean of each variable's own first lag: `1` = levels/random walk,
  `0` = white noise/differences. A scalar applied to all variables, or a
  length-`nvar` vector to set it per variable (e.g. `c(1,1,0,1,1,1,1,0)` to center
  rates on white noise and the rest on a random walk).
- `phi_mode` — pandemic-dummy shrinkage: `"single"` (one `phi` on a grid),
  `"perperiod"` (`phi_1..phi_h` by the optimiser), or a fixed scalar
  (`~0` = uninformative).
- `lambda_mode` — Minnesota tightness: a scalar (`0.2` default) or `"optimal"`.
- `tau_mode` — sum-of-coefficients tightness: a scalar, `"10lambda"`
  (default, `tau = 10*lambda`), or `"optimal"`; set `0` to drop the prior.
- `coper_mode` — co-persistence / dummy-initial-observation prior (Sims–Zha 1998),
  a single extra dummy from the average of the initial `nAR` observations that
  favours a shared stochastic trend: `0` = off (paper default), a positive scalar
  for a fixed tightness (smaller = tighter), or `"optimal"`.

**How the free hyperparameters are searched.** For `phi_mode="single"` every
`"optimal"` hyperparameter is grid-searched (same grid as `phi`); for
`"perperiod"` the free `lambda`/`tau`/`coper` join the continuous optimiser. When
`lambda` is `"optimal"` and `tau` is `"10lambda"`, `tau` tracks `10*lambda`
automatically. `"optimal"` `coper` searches a positive tightness (never 0/off) —
leave `coper_mode=0` to keep the co-persistence prior switched off.

## Correspondence with the MATLAB version

The deterministic quantities match to numerical tolerance: for the eight-variable
example, the selected `phi* = [0.155, 0.022, 0.068, 0.059, 0.170, 0.217]` and the
log marginal likelihood at those values coincide with MATLAB and Python. The
posterior median impulse responses and forecasts match in distribution; the
credible *bands* are not draw-for-draw identical across languages, because the
random-number stream differs. Seeding (`seed` in the driver) reproduces a given
R run.

## Citation

Cascaldi-Garcia, D., "Pandemic Priors."
Use of the code for research purposes is permitted with proper reference.
