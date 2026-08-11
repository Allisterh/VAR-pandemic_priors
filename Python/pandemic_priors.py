"""
Pandemic Priors -- Python implementation (external release)

Port of the MATLAB replication code for Cascaldi-Garcia, D., "Pandemic Priors."
Standard libraries only: numpy, scipy, pandas.

The module mirrors the MATLAB core one-to-one:

    pandemicpriors(...)  -> build the dummy-observation prior matrices (xd, yd)
    pp_logml(...)        -> fast closed-form log marginal likelihood (phi search)
    pp_select(...)       -> choose phi (and optionally lambda/tau/coper) by max ML
    pp_draws(...)        -> posterior draws + forecast + recursive IRF (single core)
    pp_iwishrnd(...)     -> inverse-Wishart draw (Bartlett; no SciPy dependency)
    pp_prctile(...)      -> percentiles matching MATLAB's default convention
    mlag2(...)           -> lag-matrix builder

Deterministic quantities (selected phi, posterior mean A_post, marginal
likelihood) match the MATLAB output to numerical tolerance. The posterior bands
match in distribution, not draw-for-draw: the RNG stream differs across
languages, so seeding reproduces a given language's own runs but not the exact
MATLAB draws.

Use of code for research purposes is permitted with proper reference.
Danilo Cascaldi-Garcia.
"""

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import gammaln


# ---------------------------------------------------------------------------
def mlag2(X, p):
    """Matrix of p lags of X (Traw x N*p); first p rows are zero. Cf. mlag2.m."""
    X = np.asarray(X, dtype=float)
    Traw, N = X.shape
    Xlag = np.zeros((Traw, N * p))
    for ii in range(1, p + 1):
        Xlag[p:Traw, N * (ii - 1):N * ii] = X[p - ii:Traw - ii, :]
    return Xlag


# ---------------------------------------------------------------------------
def logmvgamma(x, d):
    """log multivariate gamma Gamma_d(x). Cf. logMvGamma.m (Michael Chen)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    s = x.shape
    x = x.reshape(1, -1)
    j = ((1 - np.arange(1, d + 1)) / 2.0).reshape(d, 1)      # (d,1)
    A = x + j                                                # (d, numel(x))
    y = d * (d - 1) / 4.0 * np.log(np.pi) + gammaln(A).sum(axis=0)
    return y.reshape(s) if len(s) else float(y)


# ---------------------------------------------------------------------------
def pandemicpriors(X, Y, Yraw, nAR, constant, delta, lam, tau, epsilon,
                   phi, covid_periods, coper=0.0):
    """
    Build the Pandemic-Priors dummy-observation matrices. Cf. pandemicpriors.m.

    Returns (Xst, Yst, xd, yd): the stacked data+dummies and the dummy blocks.
    delta may be a scalar or a length-n array. coper=0 switches off the
    co-persistence (Sims-Zha) prior.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float); Yraw = np.asarray(Yraw, float)
    T = X.shape[0]; Traw = T + nAR; n = Y.shape[1]
    mu = Y.mean(axis=0)                                   # BGR full-sample mean

    phi = np.atleast_1d(np.asarray(phi, float))
    if phi.size == 1:
        phi = np.full(covid_periods, float(phi))

    delta = np.atleast_1d(np.asarray(delta, float))
    if delta.size == 1:
        delta = np.full(n, float(delta))

    # --- univariate AR(nAR) per variable -> sigma (Minnesota scale factors) ---
    ResidAR = np.full((Traw - nAR, n), np.nan)
    for i in range(n):
        YlagAR = np.full((Traw, nAR), np.nan)
        for ii in range(1, nAR + 1):
            YlagAR[nAR:Traw, ii - 1] = Yraw[nAR - ii:Traw - ii, i]
        X_AR = np.column_stack([YlagAR[nAR:Traw, :], np.ones(Traw - nAR)])
        Y_AR = Yraw[nAR:Traw, i]
        A_AR, *_ = np.linalg.lstsq(X_AR, Y_AR, rcond=None)
        ResidAR[:, i] = Y_AR - X_AR @ A_AR
    SIGMA_AR = (ResidAR.T @ ResidAR) / ResidAR.shape[0]
    sigma = np.sqrt(np.diag(SIGMA_AR))                    # length n

    yd1 = yd2 = yd3 = np.empty((0, n))
    xd_cols = n * nAR + covid_periods + (1 if constant == 1 else 0)
    xd1 = xd2 = xd3 = np.empty((0, xd_cols))

    if lam > 0:
        jp = np.diag(np.arange(1, nAR + 1).astype(float))     # diag(1..p)
        bb = np.diag(sigma * delta / lam)                      # n x n
        dd = np.diag(sigma)                                    # n x n
        ff = np.diag(sigma / lam)                              # n x n
        kron_jp_ff = np.kron(jp, ff)                           # (n*nAR) x (n*nAR)
        if constant == 1:
            yd1 = np.vstack([bb,
                             np.zeros((n * (nAR - 1), n)),
                             dd,
                             np.zeros((covid_periods + 1, n))])
            top = np.hstack([kron_jp_ff, np.zeros((n * nAR, covid_periods + 1))])
            mid = np.zeros((n, n * nAR + covid_periods + 1))
            crow = np.hstack([np.zeros((1, n * nAR)), [[epsilon]],
                              np.zeros((1, covid_periods))])
            drows = np.hstack([np.zeros((covid_periods, n * nAR + 1)),
                               np.diag(phi)])
            xd1 = np.vstack([top, mid, crow, drows])
        else:
            yd1 = np.vstack([bb,
                             np.zeros((n * (nAR - 1), n)),
                             dd,
                             np.zeros((covid_periods, n))])
            top = np.hstack([kron_jp_ff, np.zeros((n * nAR, covid_periods))])
            mid = np.zeros((n, n * nAR + covid_periods))
            drows = np.hstack([np.zeros((covid_periods, n * nAR)), np.diag(phi)])
            xd1 = np.vstack([top, mid, drows])

        # --- sum-of-coefficients prior ---
        if tau > 0:
            bb2 = np.diag(delta * mu / tau)                    # n x n
            yd2 = bb2
            kron_vec = np.kron(np.ones((1, nAR)), bb2)         # n x (n*nAR)
            pad = covid_periods + (1 if constant == 1 else 0)
            xd2 = np.hstack([kron_vec, np.zeros((n, pad))])

    # --- co-persistence / dummy-initial-observation prior (Sims-Zha) ---
    if coper > 0:
        y0 = Yraw[:nAR, :].mean(axis=0)                        # avg initial conditions
        yd3 = (y0 / coper).reshape(1, n)
        kron_y = np.kron(np.ones((1, nAR)), yd3)               # 1 x (n*nAR)
        if constant == 1:
            xd3 = np.hstack([kron_y, [[1.0 / coper]], np.zeros((1, covid_periods))])
        else:
            xd3 = np.hstack([kron_y, np.zeros((1, covid_periods))])

    yd = np.vstack([b for b in (yd1, yd2, yd3) if b.size or b.shape[0]])
    xd = np.vstack([b for b in (xd1, xd2, xd3) if b.size or b.shape[0]])
    Yst = np.vstack([Y, yd])
    Xst = np.vstack([X, xd])
    return Xst, Yst, xd, yd


# ---------------------------------------------------------------------------
def pp_logml(X, Y, Yraw, nAR, constant, delta, lam, tau, epsilon,
             phi, covid_periods, coper=0.0):
    """Fast closed-form log marginal likelihood. Cf. pp_logml.m."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    _, _, xd, yd = pandemicpriors(X, Y, Yraw, nAR, constant, delta, lam, tau,
                                  epsilon, phi, covid_periods, coper)
    T = X.shape[0]; n = Y.shape[1]
    v0 = n + 2; v1 = v0 + T

    xx0 = xd.T @ xd
    Cxx0 = cholesky(xx0, lower=True)
    b0 = solve_triangular(Cxx0.T, solve_triangular(Cxx0, xd.T @ yd, lower=True),
                          lower=False)
    e0 = yd - xd @ b0
    sigma0 = e0.T @ e0

    XXst = xx0 + X.T @ X
    CXX = cholesky(XXst, lower=True)
    XYst = xd.T @ yd + X.T @ Y
    Apo = solve_triangular(CXX.T, solve_triangular(CXX, XYst, lower=True),
                           lower=False)

    YYst = yd.T @ yd + Y.T @ Y
    S1 = YYst - XYst.T @ Apo
    S1 = (S1 + S1.T) / 2.0
    CS1 = cholesky(S1, lower=True)
    Cs0 = cholesky(sigma0, lower=True)

    ldet_xx0 = 2 * np.sum(np.log(np.diag(Cxx0)))
    ldet_XXst = 2 * np.sum(np.log(np.diag(CXX)))
    ldet_sigma0 = 2 * np.sum(np.log(np.diag(Cs0)))
    ldet_S1 = 2 * np.sum(np.log(np.diag(CS1)))

    r1 = logmvgamma(v0 / 2.0, n)
    r2 = logmvgamma(v1 / 2.0, n)

    py = (-(T * n / 2.0) * np.log(np.pi)
          + (n / 2.0) * (ldet_xx0 - ldet_XXst)
          + (v0 / 2.0) * ldet_sigma0
          + (r2 - r1)
          - (v1 / 2.0) * ldet_S1)
    return float(py)


# ---------------------------------------------------------------------------
def pp_select(phi_mode, lambda_mode, tau_mode,
              X, Y, Yraw, nAR, constant, delta, epsilon, covid_periods, phi_grid,
              coper_mode=0.0, verbose=True):
    """
    Choose (phi, lambda, tau, coper) by maximizing the marginal likelihood.
    Cf. pp_select.m.

    phi_mode    : 'single' (grid), 'perperiod' (optimiser), or a fixed scalar/vec
    lambda_mode : scalar, or 'optimal'
    tau_mode    : scalar, '10lambda' (tau = 10*lambda), or 'optimal'
    coper_mode  : scalar (0 = off), or 'optimal'
    Returns (phi_use, lam, tau, coper).
    """
    from scipy.optimize import fmin

    def is_str(s): return isinstance(s, str)
    lam_opt = is_str(lambda_mode) and lambda_mode.lower() == 'optimal'
    tau_opt = is_str(tau_mode) and tau_mode.lower() == 'optimal'
    tau_tie = is_str(tau_mode) and tau_mode.lower() in ('10lambda', '10*lambda')
    cop_opt = is_str(coper_mode) and coper_mode.lower() == 'optimal'

    lambda_fix = 0.0 if lam_opt else float(lambda_mode)
    tau_fix = 0.0 if (tau_opt or tau_tie) else float(tau_mode)
    coper_fix = 0.0 if cop_opt else float(coper_mode)

    phi_grid = np.asarray(phi_grid, float)
    lb, ub = phi_grid.min(), phi_grid.max()
    g = lambda z: lb + (ub - lb) / (1.0 + np.exp(-z))       # (lb,ub)
    ginv = lambda pv: np.log((pv - lb) / (ub - pv))

    is_perperiod = is_str(phi_mode) and phi_mode.lower() == 'perperiod'
    is_single = is_str(phi_mode) and phi_mode.lower() == 'single'

    if is_perperiod:
        h = covid_periods
        idx_lam = idx_tau = idx_cop = -1
        npar = h
        if lam_opt: idx_lam = npar; npar += 1
        if tau_opt: idx_tau = npar; npar += 1
        if cop_opt: idx_cop = npar; npar += 1

        def negML(z):
            z = np.atleast_1d(z)
            phi = g(z[:h])
            lam = g(z[idx_lam]) if lam_opt else lambda_fix
            if tau_opt:      ta = g(z[idx_tau])
            elif tau_tie:    ta = 10.0 * lam
            else:            ta = tau_fix
            cp = g(z[idx_cop]) if cop_opt else coper_fix
            return -pp_logml(X, Y, Yraw, nAR, constant, delta, lam, ta,
                             epsilon, phi, covid_periods, cp)

        best = np.inf; zbest = None
        for s0 in (0.3, 1.0, 1.7):                          # multi-start
            z0 = np.full(npar, ginv(s0))
            if idx_lam >= 0: z0[idx_lam] = ginv(0.2)
            if idx_tau >= 0: z0[idx_tau] = ginv(min(2.0, ub))
            if idx_cop >= 0: z0[idx_cop] = ginv(1.0)
            zh = fmin(negML, z0, disp=False, maxfun=4000, maxiter=4000,
                      xtol=1e-6, ftol=1e-6)
            f = negML(zh)
            if f < best: best = f; zbest = zh
        zbest = np.atleast_1d(zbest)
        phi_use = g(zbest[:h])
        lam = g(zbest[idx_lam]) if lam_opt else lambda_fix
        if tau_opt:   tau = g(zbest[idx_tau])
        elif tau_tie: tau = 10.0 * lam
        else:         tau = tau_fix
        coper = g(zbest[idx_cop]) if cop_opt else coper_fix

    else:
        n_extra_opt = int(lam_opt) + int(tau_opt) + int(cop_opt)
        if n_extra_opt >= 2:
            import warnings
            warnings.warn("Grid search with %d 'optimal' hyperparameters (+phi) "
                          "multiplies the number of marginal-likelihood "
                          "evaluations; consider phi_mode='perperiod'."
                          % n_extra_opt)
        phi_cands = list(phi_grid) if is_single else [phi_mode]
        lam_vals = phi_grid if lam_opt else [lambda_fix]
        cop_vals = phi_grid if cop_opt else [coper_fix]

        best = -np.inf
        phi_use, lam, tau, coper = phi_cands[0], lam_vals[0], np.nan, cop_vals[0]
        for lamv in lam_vals:
            if tau_tie:   tau_vals = [10.0 * lamv]
            elif tau_opt: tau_vals = phi_grid
            else:         tau_vals = [tau_fix]
            for tav in tau_vals:
                for cpv in cop_vals:
                    for phv in phi_cands:
                        ml = pp_logml(X, Y, Yraw, nAR, constant, delta, lamv, tav,
                                      epsilon, phv, covid_periods, cpv)
                        if ml > best:
                            best = ml
                            phi_use, lam, tau, coper = phv, lamv, tav, cpv

    if verbose:
        if np.ndim(phi_use) == 0 or np.size(phi_use) == 1:
            phistr = "%.3f" % float(np.ravel(phi_use)[0])
        else:
            phistr = "[" + ", ".join("%.3f" % v for v in np.ravel(phi_use)) + "]"
        print("Selected phi = %s | lambda = %.3f | tau = %.3f | coper = %.3f"
              % (phistr, lam, tau, coper))
    return phi_use, lam, tau, coper


# ---------------------------------------------------------------------------
def _randgamma(a, rng):
    """Gamma(shape=a, scale=1), Marsaglia-Tsang. Cf. randgamma in pp_iwishrnd.m."""
    if a < 1:
        return _randgamma(a + 1, rng) * rng.random() ** (1.0 / a)
    d = a - 1.0 / 3.0
    c = 1.0 / np.sqrt(9.0 * d)
    while True:
        x = rng.standard_normal()
        v = (1.0 + c * x) ** 3
        if v <= 0:
            continue
        u = rng.random()
        if np.log(u) < 0.5 * x * x + d - d * v + d * np.log(v):
            return d * v


def pp_iwishrnd(Tau, df, C=None, rng=None):
    """Inverse-Wishart draw via Bartlett decomposition. Cf. pp_iwishrnd.m."""
    if rng is None:
        rng = np.random.default_rng()
    n = Tau.shape[0]
    if C is None:
        C = cholesky(Tau, lower=True)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = np.sqrt(2.0 * _randgamma((df - i) / 2.0, rng))   # chi2_{df-i}
    if n > 1:
        idx = np.tril_indices(n, -1)
        A[idx] = rng.standard_normal(len(idx[0]))
    K = solve_triangular(A, C.T, lower=True, trans='T').T          # C * inv(A')
    Sigma = K @ K.T
    return (Sigma + Sigma.T) / 2.0


# ---------------------------------------------------------------------------
def pp_prctile(X, p, axis=0):
    """Percentiles matching MATLAB's default (plotting-position) convention.
    Cf. pp_prctile.m. p in [0,100]."""
    X = np.asarray(X, float)
    p = np.atleast_1d(np.asarray(p, float))
    Xs = np.sort(X, axis=axis)
    N = Xs.shape[axis]
    if N == 1:
        rep = np.repeat(np.take(Xs, 0, axis=axis)[np.newaxis, ...], p.size, axis=0)
        return rep
    pos = 100.0 * (np.arange(1, N + 1) - 0.5) / N
    Xm = np.moveaxis(Xs, axis, 0).reshape(N, -1)
    Q = np.empty((p.size, Xm.shape[1]))
    for c in range(Xm.shape[1]):
        Q[:, c] = np.interp(p, pos, Xm[:, c])              # clamps outside range
    out_shape = list(np.moveaxis(Xs, axis, 0).shape)
    out_shape[0] = p.size
    Q = Q.reshape(out_shape)
    return np.moveaxis(Q, 0, axis)


# ---------------------------------------------------------------------------
def pp_draws(X, Y, xd, yd, nAR, covid_periods, nimp, shocks, rps,
             seed=None, screen=True):
    """
    Posterior draws + forecast + recursive (Cholesky) IRF. Cf. pp_draws.m.

    Returns a dict with keys: A_post, SSE_post, v1, irf, fcst, C_auto,
    C_dummies, discarded, shocks.
      irf   : (n, rps, nimp, n)   -- shock, draw, horizon, variable
      fcst  : (rps, n, nimp)
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    rng = np.random.default_rng(seed)
    T, k = X.shape
    n = Y.shape[1]; nvar = n
    sizeCF = nAR * nvar

    shocks_arr = np.atleast_1d(np.asarray(shocks))
    if shocks_arr.size == 1:
        shock_idx = np.arange(int(shocks_arr[0]))          # 0-based: first k
    else:
        shock_idx = np.where(shocks_arr.ravel() != 0)[0]

    Xst = np.vstack([X, xd]); Yst = np.vstack([Y, yd])
    XXst = Xst.T @ Xst
    C = cholesky(XXst, lower=True)
    XYst = Xst.T @ Yst
    A_post = solve_triangular(C.T, solve_triangular(C, XYst, lower=True), lower=False)
    RESID = Yst - Xst @ A_post
    SSE_post = RESID.T @ RESID
    SSE_post = (SSE_post + SSE_post.T) / 2.0
    v1 = Xst.shape[0] + 2 - Xst.shape[1]
    CSSE = cholesky(SSE_post, lower=True)

    irf = np.zeros((n, rps, nimp, n))
    fcst = np.zeros((rps, n, nimp))
    C_auto = np.zeros((rps, n, n))
    C_dummies = np.zeros((rps, 1 + covid_periods, n))
    discarded = 0

    Ylast = Y[-1, :].copy()
    # lags 1..p-1 + constant/covid columns (drop the oldest lag block)
    Xtail = np.concatenate([X[-1, :nvar * (nAR - 1)], X[-1, nvar * nAR:]])

    Acomp = np.zeros((nvar * nAR, nvar * nAR))
    Acomp[nvar:nvar * nAR, :nvar * nAR - nvar] = np.eye(nvar * nAR - nvar)

    for iii in range(rps):
        while True:
            sigmarep = pp_iwishrnd(SSE_post, v1, CSSE, rng)
            # matrix-normal: A_post + (C')^{-1} randn(k,n) chol(sigmarep, upper)
            Z = rng.standard_normal((k, n))
            nbeta = A_post + solve_triangular(C.T, Z, lower=False) @ cholesky(sigmarep, lower=False)
            Acomp[:nvar, :] = nbeta[:nvar * nAR, :].T
            if not screen:
                break
            roots = np.sort(np.abs(np.linalg.eigvals(Acomp)))
            if roots[sizeCF - 1] < 1.01:
                break
            discarded += 1
        A0hat = cholesky(sigmarep, lower=False).T          # lower Cholesky, EBP first

        C_auto[iii, :, :] = nbeta[:nvar, :nvar]
        C_dummies[iii, :, :] = nbeta[nvar * nAR:, :]

        # forecast
        Y_for = Ylast.copy()
        X_for = np.concatenate([Ylast, Xtail])
        for ww in range(nimp):
            if ww > 0:
                Y_for = X_for @ nbeta
                X_for = np.concatenate([Y_for, X_for[:nvar * (nAR - 1)], X_for[nvar * nAR:]])
            fcst[iii, :, ww] = Y_for

        # IRFs: propagate companion state
        for r in shock_idx:
            imp = np.zeros((nimp, nvar))
            z = np.concatenate([A0hat[:, r], np.zeros(nvar * nAR - nvar)])
            imp[0, :] = z[:nvar]
            for kk in range(1, nimp):
                z = Acomp @ z
                imp[kk, :] = z[:nvar]
            if imp[0, r] < 0:
                imp = -imp
            irf[r, iii, :, :] = imp

    return dict(A_post=A_post, SSE_post=SSE_post, v1=v1, irf=irf, fcst=fcst,
                C_auto=C_auto, C_dummies=C_dummies, discarded=discarded,
                shocks=shock_idx)
