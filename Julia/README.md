# Pandemic Priors — Julia implementation

Julia port of the Pandemic Priors replication code, from Cascaldi-Garcia, D.,
"Pandemic Priors." A one-to-one translation of the MATLAB version. The core uses
only the standard library plus `SpecialFunctions` and `Optim`.

Because Julia allows Unicode identifiers, the code uses the paper's own Greek
notation, so the mathematical parameters read as in the manuscript:

| symbol | meaning |
|--------|---------|
| `λ` | overall (Minnesota) tightness |
| `τ` | sum-of-coefficients tightness |
| `ϕ` | pandemic-dummy shrinkage (scalar or `ϕ_1..ϕ_h`) |
| `δ` | prior mean of the own first lag |
| `ϵ` | diffuse intercept prior |
| `γc` | co-persistence (dummy-initial-observation) tightness |
| `σ`, `Σ` | Minnesota scale factors / residual covariance |

## Requirements

- Julia 1.6+
- `Optim` (hyperparameter search), `XLSX` (read the data), `Plots` (figures)

```julia
import Pkg; Pkg.add(["Optim", "XLSX", "Plots"])
```

## How to run

```
julia run_pandemic_priors.jl
```

Options are in the `USER SETTINGS` block at the top of `run_pandemic_priors.jl`.

## Files

```
PandemicPriors.jl        core module (include-d by the driver)
run_pandemic_priors.jl   driver: build the BVAR, select ϕ, draw, plot
data/Data_20260524.xlsx  example dataset (Jan 1975 – Dec 2025, 8 monthly series)
```

`PandemicPriors.jl` exports `mlag2`, `pandemicpriors`, `pp_logml`, `pp_select`,
`pp_draws`, `pp_iwishrnd`, and `pp_prctile`, mirroring the MATLAB functions.

## Settings (in `run_pandemic_priors.jl`)

All options live in the `USER SETTINGS` block, with the same inline notes.

Model / sampling:

- `nAR` — VAR lags; `covid_periods` — number of pandemic dummy months `h`;
  `nimp` — IRF/forecast horizon; `rps` — posterior draws; `constant` — include intercept.
- `shocks` — which recursive (Cholesky) shocks to compute: a count (`1` = first
  shock, EBP) or a 0/1 selector over the variables (e.g. `[1,0,0,1,0,0,0,0]` =
  shocks 1 & 4). Only selected shocks are computed.
- `ϵ` — diffuse prior on the constant.
- `seed` — rng seed for reproducible bands (`nothing` to skip).
- `screen` — stationarity screen: `true` (default) rejects draws with an explosive
  companion root (`max|eig| >= 1.01`); `false` keeps every draw (faster, but the
  retained explosive draws fatten the tails).

Prior / shrinkage:

- `δ` — prior mean of each variable's own first lag: `1` = levels/random walk,
  `0` = white noise/differences. A scalar applied to all variables, or a
  length-`nvar` vector to set it per variable (e.g. `[1,1,0,1,1,1,1,0]` to center
  rates on white noise and the rest on a random walk).
- `ϕ_mode` — pandemic-dummy shrinkage: `"single"` (one `ϕ` on a grid),
  `"perperiod"` (`ϕ_1..ϕ_h` by the optimiser), or a fixed scalar
  (`~0` = uninformative).
- `λ_mode` — Minnesota tightness: a scalar (`0.2` default) or `"optimal"`.
- `τ_mode` — sum-of-coefficients tightness: a scalar, `"10lambda"`
  (default, `τ = 10λ`), or `"optimal"`; set `0` to drop the prior.
- `γc_mode` — co-persistence / dummy-initial-observation prior (Sims–Zha 1998),
  a single extra dummy from the average of the initial `nAR` observations that
  favours a shared stochastic trend: `0` = off (paper default), a positive scalar
  for a fixed tightness (smaller = tighter), or `"optimal"`.

**How the free hyperparameters are searched.** For `ϕ_mode="single"` every
`"optimal"` hyperparameter is grid-searched (same grid as `ϕ`); for `"perperiod"`
the free `λ`/`τ`/`γc` join the continuous optimiser. When `λ` is `"optimal"` and
`τ` is `"10lambda"`, `τ` tracks `10λ` automatically. `"optimal"` `γc` searches a
positive tightness (never 0/off) — leave `γc_mode=0` to keep the co-persistence
prior switched off.

## Correspondence with the MATLAB version

The deterministic quantities match to numerical tolerance: for the eight-variable
example, the selected `ϕ* = [0.155, 0.022, 0.068, 0.059, 0.170, 0.217]` and the
log marginal likelihood at those values coincide with MATLAB and Python. The
posterior median impulse responses and forecasts match in distribution; the
credible *bands* are not draw-for-draw identical across languages, because the
random-number stream differs. Seeding (`seed` in the driver) reproduces a given
Julia run.

## Citation

Cascaldi-Garcia, D., "Pandemic Priors."
Use of the code for research purposes is permitted with proper reference.
