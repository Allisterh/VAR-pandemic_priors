# Pandemic Priors -- R implementation (external release)
#
# Port of the MATLAB replication code for Cascaldi-Garcia, D., "Pandemic Priors."
# Base R only (no extra packages for the core; the driver uses readxl for the
# xlsx and base graphics for the figures).
#
# Mirrors the MATLAB / Python core one-to-one:
#   pandemicpriors(...)  build the dummy-observation prior matrices (xd, yd)
#   pp_logml(...)        fast closed-form log marginal likelihood (phi search)
#   pp_select(...)       choose phi (and optionally lambda/tau/coper) by max ML
#   pp_draws(...)        posterior draws + forecast + recursive IRF (single core)
#   pp_iwishrnd(...)     inverse-Wishart draw (Bartlett; base R)
#   pp_prctile(...)      percentiles matching MATLAB's default convention
#   mlag2(...)           lag-matrix builder
#
# Deterministic quantities (selected phi, A_post, marginal likelihood) match the
# MATLAB/Python output to numerical tolerance. Posterior bands match in
# distribution, not draw-for-draw (RNG stream differs across languages).
#
# Use of code for research purposes is permitted with proper reference.
# Danilo Cascaldi-Garcia.

# ---------------------------------------------------------------------------
mlag2 <- function(X, p) {
  X <- as.matrix(X); Traw <- nrow(X); N <- ncol(X)
  Xlag <- matrix(0, Traw, N * p)
  for (ii in 1:p)
    Xlag[(p + 1):Traw, (N * (ii - 1) + 1):(N * ii)] <- X[(p + 1 - ii):(Traw - ii), , drop = FALSE]
  Xlag
}

# ---------------------------------------------------------------------------
logmvgamma <- function(x, d) {
  # log multivariate gamma Gamma_d(x); x scalar here
  j <- (1 - (1:d)) / 2
  d * (d - 1) / 4 * log(pi) + sum(lgamma(x + j))
}

# ---------------------------------------------------------------------------
pandemicpriors <- function(X, Y, Yraw, nAR, constant, delta, lambda, tau,
                           epsilon, phi, covid_periods, coper = 0) {
  X <- as.matrix(X); Y <- as.matrix(Y); Yraw <- as.matrix(Yraw)
  T <- nrow(X); Traw <- T + nAR; n <- ncol(Y)
  mu <- colMeans(Y)

  if (length(phi) == 1) phi <- rep(phi, covid_periods)
  if (length(delta) == 1) delta <- rep(delta, n)

  # univariate AR(nAR) per variable -> sigma
  ResidAR <- matrix(NA, Traw - nAR, n)
  for (i in 1:n) {
    YlagAR <- matrix(NA, Traw, nAR)
    for (ii in 1:nAR)
      YlagAR[(nAR + 1):Traw, ii] <- Yraw[(nAR + 1 - ii):(Traw - ii), i]
    X_AR <- cbind(YlagAR[(nAR + 1):Traw, , drop = FALSE], 1)
    Y_AR <- Yraw[(nAR + 1):Traw, i]
    A_AR <- solve(t(X_AR) %*% X_AR, t(X_AR) %*% Y_AR)
    ResidAR[, i] <- Y_AR - X_AR %*% A_AR
  }
  SIGMA_AR <- (t(ResidAR) %*% ResidAR) / nrow(ResidAR)
  sigma <- sqrt(diag(SIGMA_AR))

  yd1 <- yd2 <- yd3 <- NULL
  xd1 <- xd2 <- xd3 <- NULL

  if (lambda > 0) {
    jp <- diag(1:nAR)
    bb <- diag(sigma * delta / lambda)
    dd <- diag(sigma)
    ff <- diag(sigma / lambda)
    kron_jp_ff <- kronecker(jp, ff)
    if (constant == 1) {
      yd1 <- rbind(bb, matrix(0, n * (nAR - 1), n), dd, matrix(0, covid_periods + 1, n))
      top <- cbind(kron_jp_ff, matrix(0, n * nAR, covid_periods + 1))
      mid <- matrix(0, n, n * nAR + covid_periods + 1)
      crow <- cbind(matrix(0, 1, n * nAR), epsilon, matrix(0, 1, covid_periods))
      drows <- cbind(matrix(0, covid_periods, n * nAR + 1), diag(phi, covid_periods))
      xd1 <- rbind(top, mid, crow, drows)
    } else {
      yd1 <- rbind(bb, matrix(0, n * (nAR - 1), n), dd, matrix(0, covid_periods, n))
      top <- cbind(kron_jp_ff, matrix(0, n * nAR, covid_periods))
      mid <- matrix(0, n, n * nAR + covid_periods)
      drows <- cbind(matrix(0, covid_periods, n * nAR), diag(phi, covid_periods))
      xd1 <- rbind(top, mid, drows)
    }
    if (tau > 0) {
      bb2 <- diag(delta * mu / tau)
      yd2 <- bb2
      kron_vec <- kronecker(matrix(1, 1, nAR), bb2)
      pad <- covid_periods + ifelse(constant == 1, 1, 0)
      xd2 <- cbind(kron_vec, matrix(0, n, pad))
    }
  }

  if (coper > 0) {
    y0 <- colMeans(Yraw[1:nAR, , drop = FALSE])
    yd3 <- matrix(y0 / coper, 1, n)
    kron_y <- kronecker(matrix(1, 1, nAR), yd3)
    if (constant == 1) {
      xd3 <- cbind(kron_y, 1 / coper, matrix(0, 1, covid_periods))
    } else {
      xd3 <- cbind(kron_y, matrix(0, 1, covid_periods))
    }
  }

  yd <- rbind(yd1, yd2, yd3)
  xd <- rbind(xd1, xd2, xd3)
  Yst <- rbind(Y, yd); Xst <- rbind(X, xd)
  list(Xst = Xst, Yst = Yst, xd = xd, yd = yd)
}

# ---------------------------------------------------------------------------
pp_logml <- function(X, Y, Yraw, nAR, constant, delta, lambda, tau, epsilon,
                     phi, covid_periods, coper = 0) {
  X <- as.matrix(X); Y <- as.matrix(Y)
  pp <- pandemicpriors(X, Y, Yraw, nAR, constant, delta, lambda, tau, epsilon,
                       phi, covid_periods, coper)
  xd <- pp$xd; yd <- pp$yd
  T <- nrow(X); n <- ncol(Y); v0 <- n + 2; v1 <- v0 + T

  xx0 <- t(xd) %*% xd
  Cxx0 <- chol(xx0)                       # upper: t(Cxx0) %*% Cxx0 = xx0
  b0 <- backsolve(Cxx0, forwardsolve(t(Cxx0), t(xd) %*% yd))
  e0 <- yd - xd %*% b0
  sigma0 <- t(e0) %*% e0

  XXst <- xx0 + t(X) %*% X
  CXX <- chol(XXst)
  XYst <- t(xd) %*% yd + t(X) %*% Y
  Apo <- backsolve(CXX, forwardsolve(t(CXX), XYst))

  YYst <- t(yd) %*% yd + t(Y) %*% Y
  S1 <- YYst - t(XYst) %*% Apo
  S1 <- (S1 + t(S1)) / 2
  CS1 <- chol(S1)
  Cs0 <- chol(sigma0)

  ldet_xx0    <- 2 * sum(log(diag(Cxx0)))
  ldet_XXst   <- 2 * sum(log(diag(CXX)))
  ldet_sigma0 <- 2 * sum(log(diag(Cs0)))
  ldet_S1     <- 2 * sum(log(diag(CS1)))

  r1 <- logmvgamma(v0 / 2, n); r2 <- logmvgamma(v1 / 2, n)
  -(T * n / 2) * log(pi) + (n / 2) * (ldet_xx0 - ldet_XXst) +
    (v0 / 2) * ldet_sigma0 + (r2 - r1) - (v1 / 2) * ldet_S1
}

# ---------------------------------------------------------------------------
pp_select <- function(phi_mode, lambda_mode, tau_mode, X, Y, Yraw, nAR, constant,
                      delta, epsilon, covid_periods, phi_grid, coper_mode = 0,
                      verbose = TRUE) {
  is_str <- function(s) is.character(s)
  lam_opt <- is_str(lambda_mode) && tolower(lambda_mode) == "optimal"
  tau_opt <- is_str(tau_mode) && tolower(tau_mode) == "optimal"
  tau_tie <- is_str(tau_mode) && tolower(tau_mode) %in% c("10lambda", "10*lambda")
  cop_opt <- is_str(coper_mode) && tolower(coper_mode) == "optimal"

  lambda_fix <- if (lam_opt) 0 else as.numeric(lambda_mode)
  tau_fix <- if (tau_opt || tau_tie) 0 else as.numeric(tau_mode)
  coper_fix <- if (cop_opt) 0 else as.numeric(coper_mode)

  lb <- min(phi_grid); ub <- max(phi_grid)
  g <- function(z) lb + (ub - lb) / (1 + exp(-z))
  ginv <- function(pv) log((pv - lb) / (ub - pv))

  is_perperiod <- is_str(phi_mode) && tolower(phi_mode) == "perperiod"
  is_single <- is_str(phi_mode) && tolower(phi_mode) == "single"

  if (is_perperiod) {
    h <- covid_periods
    idx_lam <- idx_tau <- idx_cop <- 0; npar <- h
    if (lam_opt) { npar <- npar + 1; idx_lam <- npar }
    if (tau_opt) { npar <- npar + 1; idx_tau <- npar }
    if (cop_opt) { npar <- npar + 1; idx_cop <- npar }

    negML <- function(z) {
      phi <- g(z[1:h])
      lam <- if (lam_opt) g(z[idx_lam]) else lambda_fix
      ta <- if (tau_opt) g(z[idx_tau]) else if (tau_tie) 10 * lam else tau_fix
      cp <- if (cop_opt) g(z[idx_cop]) else coper_fix
      -pp_logml(X, Y, Yraw, nAR, constant, delta, lam, ta, epsilon, phi,
                covid_periods, cp)
    }
    # helper: run Nelder-Mead, then RESTART from its own solution until it stops
    # improving. R's Nelder-Mead tends to stop early on this 6-D transformed
    # surface; restarting from the converged simplex reliably reaches the optimum.
    run_nm <- function(z0) {
      val <- Inf; z <- z0
      repeat {
        opt <- optim(z, negML, method = "Nelder-Mead",
                     control = list(maxit = 4000, reltol = 1e-10))
        if (opt$value > val - 1e-8) { val <- min(val, opt$value); z <- opt$par; break }
        val <- opt$value; z <- opt$par
      }
      list(value = val, par = z)
    }

    best <- Inf; zbest <- NULL
    starts <- c(0.3, 1.0, 1.7)                  # same spread starts as Python/Julia
    for (s0 in starts) {
      z0 <- rep(ginv(min(max(s0, lb + 1e-6), ub - 1e-6)), npar)
      if (idx_lam > 0) z0[idx_lam] <- ginv(0.2)
      if (idx_tau > 0) z0[idx_tau] <- ginv(min(2, ub))
      if (idx_cop > 0) z0[idx_cop] <- ginv(1)
      opt <- run_nm(z0)
      if (opt$value < best) { best <- opt$value; zbest <- opt$par }
    }
    phi_use <- g(zbest[1:h])
    lam <- if (lam_opt) g(zbest[idx_lam]) else lambda_fix
    tau <- if (tau_opt) g(zbest[idx_tau]) else if (tau_tie) 10 * lam else tau_fix
    coper <- if (cop_opt) g(zbest[idx_cop]) else coper_fix
  } else {
    n_extra_opt <- sum(lam_opt, tau_opt, cop_opt)
    if (n_extra_opt >= 2)
      warning("Grid search with multiple 'optimal' hyperparameters multiplies ",
              "the number of ML evaluations; consider phi_mode='perperiod'.")
    phi_cands <- if (is_single) as.list(phi_grid) else list(phi_mode)
    lam_vals <- if (lam_opt) phi_grid else lambda_fix
    cop_vals <- if (cop_opt) phi_grid else coper_fix
    best <- -Inf
    phi_use <- phi_cands[[1]]; lam <- lam_vals[1]; tau <- NA; coper <- cop_vals[1]
    for (lamv in lam_vals) {
      tau_vals <- if (tau_tie) 10 * lamv else if (tau_opt) phi_grid else tau_fix
      for (tav in tau_vals) for (cpv in cop_vals) for (phv in phi_cands) {
        ml <- pp_logml(X, Y, Yraw, nAR, constant, delta, lamv, tav, epsilon,
                       phv, covid_periods, cpv)
        if (ml > best) { best <- ml; phi_use <- phv; lam <- lamv; tau <- tav; coper <- cpv }
      }
    }
  }
  if (verbose) {
    phistr <- if (length(phi_use) == 1) sprintf("%.3f", phi_use)
              else paste0("[", paste(sprintf("%.3f", phi_use), collapse = ", "), "]")
    cat(sprintf("Selected phi = %s | lambda = %.3f | tau = %.3f | coper = %.3f\n",
                phistr, lam, tau, coper))
  }
  list(phi = phi_use, lambda = lam, tau = tau, coper = coper)
}

# ---------------------------------------------------------------------------
.randgamma <- function(a) {
  if (a < 1) return(.randgamma(a + 1) * runif(1)^(1 / a))
  d <- a - 1 / 3; c <- 1 / sqrt(9 * d)
  repeat {
    x <- rnorm(1); v <- (1 + c * x)^3
    if (v <= 0) next
    u <- runif(1)
    if (log(u) < 0.5 * x^2 + d - d * v + d * log(v)) return(d * v)
  }
}

pp_iwishrnd <- function(Tau, df, C = NULL) {
  n <- nrow(Tau)
  if (is.null(C)) C <- t(chol(Tau))            # lower: C %*% t(C) = Tau
  A <- matrix(0, n, n)
  for (i in 1:n) A[i, i] <- sqrt(2 * .randgamma((df - (i - 1)) / 2))
  if (n > 1) for (i in 2:n) for (j in 1:(i - 1)) A[i, j] <- rnorm(1)
  K <- t(solve(t(A), t(C)))                    # C %*% inv(A')  == solve(A) applied
  Sigma <- K %*% t(K)
  (Sigma + t(Sigma)) / 2
}

# ---------------------------------------------------------------------------
pp_prctile <- function(X, p) {
  # percentiles down columns (dim 1), MATLAB plotting-position convention
  X <- as.matrix(X); N <- nrow(X); m <- ncol(X)
  if (N == 1) return(matrix(rep(X, each = length(p)), length(p), m))
  pos <- 100 * ((1:N) - 0.5) / N
  Q <- matrix(0, length(p), m)
  for (c in 1:m) Q[, c] <- approx(pos, sort(X[, c]), xout = p, rule = 2)$y
  Q
}

# ---------------------------------------------------------------------------
pp_draws <- function(X, Y, xd, yd, nAR, covid_periods, nimp, shocks, rps,
                     seed = NULL, screen = TRUE) {
  X <- as.matrix(X); Y <- as.matrix(Y)
  if (!is.null(seed)) set.seed(seed)
  T <- nrow(X); k <- ncol(X); n <- ncol(Y); nvar <- n
  sizeCF <- nAR * nvar

  if (length(shocks) == 1) shock_idx <- 1:shocks else shock_idx <- which(shocks != 0)

  Xst <- rbind(X, xd); Yst <- rbind(Y, yd)
  XXst <- t(Xst) %*% Xst
  C <- t(chol(XXst))                            # lower
  XYst <- t(Xst) %*% Yst
  A_post <- backsolve(t(C), forwardsolve(C, XYst))
  RESID <- Yst - Xst %*% A_post
  SSE_post <- t(RESID) %*% RESID; SSE_post <- (SSE_post + t(SSE_post)) / 2
  v1 <- nrow(Xst) + 2 - ncol(Xst)
  CSSE <- t(chol(SSE_post))

  irf <- array(0, c(n, rps, nimp, n))
  fcst <- array(0, c(rps, n, nimp))
  C_auto <- array(0, c(rps, n, n))
  C_dummies <- array(0, c(rps, 1 + covid_periods, n))
  discarded <- 0

  Ylast <- Y[T, ]
  Xtail <- c(X[T, 1:(nvar * (nAR - 1))], X[T, (nvar * nAR + 1):k])

  Acomp <- matrix(0, nvar * nAR, nvar * nAR)
  Acomp[(nvar + 1):(nvar * nAR), 1:(nvar * nAR - nvar)] <- diag(nvar * nAR - nvar)

  for (iii in 1:rps) {
    repeat {
      sigmarep <- pp_iwishrnd(SSE_post, v1, CSSE)
      Z <- matrix(rnorm(k * n), k, n)
      nbeta <- A_post + backsolve(t(C), Z) %*% chol(sigmarep)   # chol upper
      Acomp[1:nvar, ] <- t(nbeta[1:(nvar * nAR), ])
      if (!screen) break
      roots <- sort(Mod(eigen(Acomp, only.values = TRUE)$values))
      if (roots[sizeCF] < 1.01) break else discarded <- discarded + 1
    }
    A0hat <- t(chol(sigmarep))                  # lower Cholesky

    C_auto[iii, , ] <- nbeta[1:nvar, 1:nvar]
    C_dummies[iii, , ] <- nbeta[(nvar * nAR + 1):nrow(nbeta), ]

    Y_for <- Ylast; X_for <- c(Ylast, Xtail)
    for (ww in 1:nimp) {
      if (ww > 1) {
        Y_for <- as.numeric(X_for %*% nbeta)
        X_for <- c(Y_for, X_for[1:(nvar * (nAR - 1))], X_for[(nvar * nAR + 1):length(X_for)])
      }
      fcst[iii, , ww] <- Y_for
    }

    for (r in shock_idx) {
      imp <- matrix(0, nimp, nvar)
      z <- c(A0hat[, r], rep(0, nvar * nAR - nvar))
      imp[1, ] <- z[1:nvar]
      for (kk in 2:nimp) { z <- Acomp %*% z; imp[kk, ] <- z[1:nvar] }
      if (imp[1, r] < 0) imp <- -imp
      irf[r, iii, , ] <- imp
    }
  }
  list(A_post = A_post, SSE_post = SSE_post, v1 = v1, irf = irf, fcst = fcst,
       C_auto = C_auto, C_dummies = C_dummies, discarded = discarded,
       shocks = shock_idx)
}
