"""
Pandemic Priors -- Python driver (external release)

Mirror of the MATLAB script run_pandemic_priors.m: builds the eight-variable
monthly BVAR, selects the shrinkage parameter phi, draws the posterior, and
produces the EBP-shock impulse responses and the forecast fan charts.

Run:  python run_pandemic_priors.py
Requires: numpy, scipy, pandas (and matplotlib for the figures).
"""

import os
import time
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # run from this script's folder

from pandemic_priors import (pandemicpriors, pp_select, pp_draws, pp_prctile,
                             mlag2)

# ============================ USER SETTINGS ==============================
nAR           = 12          # VAR lags
covid_periods = 6           # pandemic dummy months, from Mar/2020 (h)
nimp          = 12          # IRF / forecast horizon (months)
rps           = 10000       # posterior draws
shocks        = 1           # recursive (Cholesky) shocks to compute. Either a count
                            # (1 = first shock, EBP) or a 0/1 selector over the
                            # variables, e.g. [1,0,0,1,0,0,0,0] = shocks 1 & 4.
                            # Only selected shocks are computed (no extra cost).
constant      = 1           # include intercept
delta         = 1.0         # prior mean of own first lag (1 = levels/random walk,
                            # 0 = white noise/differences). A SCALAR applied to all
                            # variables, OR a length-nvar array to set it per
                            # variable, e.g. [1,1,0,1,1,1,1,0] to center rates on
                            # white noise and the rest on a random walk. It enters
                            # element-wise in the Minnesota block (sigma*delta) and
                            # the sum-of-coefficients block (delta*mu).
epsilon       = 0.001       # diffuse prior on the constant
seed          = 1           # rng seed for reproducible bands (None to skip)
screen        = True        # stationarity screen: reject draws with an explosive
                            # companion root (max|eig| >= 1.01).
                            #   True  -> default
                            #   False -> keep every draw; faster loop, but retained
                            #            explosive draws fatten the tails.

# --- how the pandemic-dummy shrinkage phi is chosen ---
phi_mode      = 'perperiod' # 'single'    : one phi, chosen on phi_grid
                            # 'perperiod' : phi_1..phi_h, chosen by the optimiser
                            # <scalar>    : a fixed phi (e.g. 0.1; ~0 = uninformative)
lambda_mode   = 0.2         # Minnesota tightness: a scalar (0.2 default) or 'optimal'
tau_mode      = '10lambda'  # sum-of-coefficients tightness: a scalar, '10lambda'
                            # (default, tau = 10*lambda), or 'optimal'. Set tau_mode=0
                            # to drop the sum-of-coefficients prior entirely.
coper_mode    = 0.0         # co-persistence / dummy-initial-observation prior
                            # (Sims-Zha 1998): a single extra dummy built from the
                            # average of the initial nAR observations, favouring a
                            # shared stochastic trend. 0 = OFF (paper default);
                            # a positive scalar sets a fixed tightness (smaller =
                            # tighter); or 'optimal' to select it by marginal
                            # likelihood alongside phi (and any optimal lambda/tau).
# Selection: for phi_mode 'single' every free hyperparameter is grid-searched (same
# grid as phi); for 'perperiod' free lambda/tau/coper join the optimiser. When
# lambda is 'optimal' and tau is '10lambda', tau tracks 10*lambda automatically.
# 'optimal' coper searches a positive tightness (never 0/off) -- leave coper_mode=0
# to keep the co-persistence prior switched off.

# Data
data_file  = './data/Data_20260524.xlsx'
log_vector = np.array([0, 1, 0, 1, 1, 1, 1, 0])   # 1 -> variable enters as 100*log
Yname = ['EBP', 'S&P 500', 'Shadow Rate', 'PCE', 'PCE Price Index',
         'Employment', 'Ind. Production', 'Unemp. Rate']
# Sample and pandemic window, set by date. Only rows in [sample_start, sample_end]
# are used, so the data file may extend past sample_end.
sample_start   = '1975-01-01'
sample_end     = '2025-12-01'
pandemic_start = '2020-03-01'                      # first pandemic dummy month

phi_grid = np.array([0.001, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25,
                     0.30, 0.35, 0.40, 0.45, 0.50, 0.75, 1, 2, 5])

# ============================ BUILD Y, X ================================
df = pd.read_excel(data_file)
dates = pd.to_datetime(df.iloc[:, 0])              # first column = monthly timestamps
keep = (dates >= sample_start) & (dates <= sample_end)
datevec = dates[keep].reset_index(drop=True)
data = df.iloc[keep.values, 1:].to_numpy().astype(float)   # drop date column, keep sample
Yraw = data.copy()
for e in range(log_vector.size):
    if log_vector[e] == 1:
        Yraw[:, e] = np.log(Yraw[:, e]) * 100.0
Traw, nvar = Yraw.shape

Ylag = mlag2(Yraw, nAR)
if constant == 1:
    X = np.column_stack([Ylag[nAR:Traw, :], np.ones(Traw - nAR)])
else:
    X = Ylag[nAR:Traw, :]
Y = Yraw[nAR:Traw, :]

# pandemic dummy columns: locate the pandemic-start month in the estimation sample
covid_ind = int(np.where(datevec == pd.Timestamp(pandemic_start))[0][0]) - nAR
X = np.column_stack([X, np.zeros((X.shape[0], covid_periods))])
X[covid_ind:covid_ind + covid_periods, -covid_periods:] = np.eye(covid_periods)

# ==================== SELECT HYPERPARAMETERS ============================
t0 = time.time()
phi_use, lam, tau, coper = pp_select(phi_mode, lambda_mode, tau_mode,
                                     X, Y, Yraw, nAR, constant, delta, epsilon,
                                     covid_periods, phi_grid, coper_mode)
t_phi = time.time() - t0

# ============================ ESTIMATE =================================
_, _, xd, yd = pandemicpriors(X, Y, Yraw, nAR, constant, delta, lam, tau,
                              epsilon, phi_use, covid_periods, coper)
t0 = time.time()
S = pp_draws(X, Y, xd, yd, nAR, covid_periods, nimp, shocks, rps, seed, screen)
t_draw = time.time() - t0

print("phi-search %.2fs | draws+fcst+IRF %.2fs | screen %s (discarded %d)"
      % (t_phi, t_draw, "ON" if screen else "OFF", S['discarded']))
print("A_post %dx%d, IRF %s, forecast %s"
      % (S['A_post'].shape[0], S['A_post'].shape[1],
         S['irf'].shape, S['fcst'].shape))

# ============================ FIGURES ==================================
try:
    import matplotlib.pyplot as plt
    bands = [50, 16, 84]
    grey = (0.70, 0.70, 0.70); red = (0.85, 0.11, 0.11)
    p_lines = nvar // 3; p_cols = int(np.ceil(nvar / p_lines))
    xax = np.arange(1, nimp + 1)

    for r in S['shocks']:
        fig, axes = plt.subplots(p_lines, p_cols, figsize=(11, 4.8))
        for uu, ax in enumerate(axes.ravel()[:nvar]):
            q = pp_prctile(S['irf'][r, :, :, uu], bands, axis=0)   # (3, nimp)
            ax.fill_between(xax, q[1], q[2], color=grey, edgecolor='none')
            ax.plot(xax, q[0], color=red, lw=2)
            ax.axhline(0, ls=':', color='k', lw=0.8)
            ax.set_title(Yname[uu], fontsize=9); ax.set_xlabel('months')
        fig.suptitle('%s shock' % Yname[r]); fig.tight_layout()
        fig.savefig('irf_shock%d.pdf' % (r + 1))

    fig, axes = plt.subplots(p_lines, p_cols, figsize=(11, 4.8))
    for uu, ax in enumerate(axes.ravel()[:nvar]):
        q = pp_prctile(S['fcst'][:, uu, :], bands, axis=0)
        ax.fill_between(xax, q[1], q[2], color=grey, edgecolor='none')
        ax.plot(xax, q[0], color=red, lw=2)
        ax.set_title(Yname[uu], fontsize=9); ax.set_xlabel('months')
    fig.suptitle('Forecast'); fig.tight_layout()
    fig.savefig('forecast.pdf')
    print("wrote irf_shock*.pdf and forecast.pdf")
except ImportError:
    print("(matplotlib not available; skipping figures)")
