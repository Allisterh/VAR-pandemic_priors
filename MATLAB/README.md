# Pandemic Priors — replication code (external release)

Minimal, single-core (no parallel toolbox) MATLAB implementation of the
Pandemic Priors, from Cascaldi-Garcia, D., "Pandemic Priors."

It estimates a conjugate Normal-Inverse-Wishart BVAR augmented with `n*h`
pandemic dummy coefficients whose Gaussian prior precision is `phi`, then
(1) selects `phi`, (2) draws the posterior and forecasts, and (3) identifies
the EBP shock (recursive/Cholesky, EBP ordered first). No toolboxes are
required (percentiles, the inverse-Wishart draw, and the marginal likelihood
are all self-contained).

## How to run

```matlab
run_pandemic_priors      % from this folder; it adds ./utils to the path
```

Everything is controlled from the `USER SETTINGS` block at the top of
`run_pandemic_priors.m`.

## Model / sampling settings

- `nAR` — VAR lags; `covid_periods` — number of pandemic dummy months `h`;
  `nimp` — IRF/forecast horizon; `rps` — posterior draws; `constant` — include intercept.
- `shocks` — which recursive (Cholesky) shocks to compute: a count (`1` = first
  shock, EBP) or a 0/1 selector over the variables (e.g. `[1 0 0 1 0 0 0 0]` =
  shocks 1 & 4). Only selected shocks are computed.
- `epsilon` — diffuse prior on the constant.
- `seed` — rng seed for reproducible bands (`[]` to skip).
- `screen` — stationarity screen: `true` (default) rejects draws with an explosive
  companion root (`max|eig| >= 1.01`); `false` keeps every draw (faster, but the
  retained explosive draws fatten the tails).

## Prior settings

- `phi_mode` — `'single'` (one `phi` chosen on a grid), `'perperiod'`
  (`phi_1..phi_h` chosen by the optimiser), or a fixed scalar/vector.
- `lambda_mode` — Minnesota tightness: a scalar (0.2 default) or `'optimal'`.
- `tau_mode` — sum-of-coefficients tightness: a scalar, `'10lambda'`
  (default, `tau = 10*lambda`), or `'optimal'`; set `0` to drop the
  sum-of-coefficients prior. (`tau = 10*lambda` is the Bańbura–Giannone–Reichlin
  2010 value.)
- `delta` — prior mean of each variable's own first lag (1 = levels/random
  walk, 0 = white noise/differences). May be a scalar applied to all variables
  or a `1 x nvar` row vector to set it per variable.
- `coper_mode` — co-persistence / dummy-initial-observation prior (Sims–Zha
  1998). A single extra dummy observation, built from the average of the initial
  `nAR` observations, that favours a shared stochastic trend across variables.
  `0` = OFF (the standard specification used in the paper); a positive scalar
  sets a fixed tightness (smaller = tighter); or `'optimal'` to select it by
  marginal likelihood alongside `phi` (and any optimal `lambda`/`tau`). When on,
  it is applied consistently in both the `phi` selection and the final
  estimation. `'optimal'` searches a positive tightness (it never returns 0/off)
  — leave `coper_mode = 0` to keep the prior switched off.

## A note on `coper` (explicit argument)

The three functions that take `coper` (`pandemicpriors`, `pp_logml`,
`pp_select`) require it as an **explicit** argument — there is no silent
default. This keeps the prior specification fully visible at every call site.
The driver `run_pandemic_priors.m` sets `coper_mode` in its settings block,
`pp_select` returns the resolved `coper` (a value if fixed, or the ML-selected
value if `'optimal'`), and that value is passed on to `pandemicpriors`. To
reproduce the paper exactly, leave `coper_mode = 0`.

Cost note: with `phi_mode = 'perperiod'` (the default), an `'optimal'` `coper`
is just one extra continuous dimension in the `fminsearch` — cheap. With
`phi_mode = 'single'` the selection is a nested grid, so making `lambda`, `tau`,
and `coper` all `'optimal'` at once multiplies the number of evaluations;
`pp_select` prints a warning and suggests `'perperiod'` in that case.

## Files

```
run_pandemic_priors.m      main script
utils/pandemicpriors.m     builds the dummy-observation prior matrices
utils/pp_select.m          hyperparameter selection (phi, lambda, tau)
utils/pp_logml.m           fast log marginal likelihood for the phi search
utils/pp_draws.m           posterior draws + forecast + IRF (single core)
utils/pp_prctile.m         percentiles (toolbox-free)
utils/pp_iwishrnd.m        inverse-Wishart draw (toolbox-free)
utils/logMvGamma.m         log multivariate gamma
utils/mlag2.m              lag-matrix builder
data/Data_20260524.xlsx    example dataset (see below)
```

## Example dataset

Eight monthly U.S. series, January 1975 – December 2025 (this vintage retrieved
May 2026). Columns of `data/Data_20260524.xlsx`, in order:

| # | Column   | Variable            | Enters as | Source |
|---|----------|---------------------|-----------|--------|
| 1 | EBP      | Excess Bond Premium | level     | Gilchrist–Zakrajšek; updated series, Board of Governors FEDS Notes: https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv |
| 2 | SP500    | S&P 500 stock index | 100·log   | FRED-MD (McCracken–Ng 2016), series "S&P 500" |
| 3 | WUXIA    | Shadow federal funds rate | level | Wu–Xia shadow rate, Federal Reserve Bank of Atlanta: https://www.atlantafed.org/research-and-data/data/wu-xia-shadow-federal-funds-rate |
| 4 | PCE      | Real personal consumption expenditures | 100·log | FRED |
| 5 | PCEPI    | PCE price index     | 100·log   | FRED |
| 6 | PAYEMS   | Total nonfarm payroll employment | 100·log | FRED |
| 7 | INDPRO   | Industrial production | 100·log | FRED |
| 8 | UNRATE   | Unemployment rate   | level     | FRED |

The `log_vector` in the script marks which columns enter as `100*log(x)`
(columns 2, 4, 5, 6, 7) versus in levels (EBP, shadow rate, unemployment rate).

## Citation

Cascaldi-Garcia, D., "Pandemic Priors."
Use of the code for research purposes is permitted with proper reference.
