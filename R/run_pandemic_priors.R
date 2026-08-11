# Pandemic Priors -- R driver (external release)
#
# Mirror of the MATLAB script run_pandemic_priors.m: builds the eight-variable
# monthly BVAR, selects phi, draws the posterior, and produces the EBP-shock
# impulse responses and forecast fan charts.
#
# Run:  Rscript run_pandemic_priors.R
# Requires: base R + readxl (for the xlsx).

# run from this script's own folder (robust to Rscript / source() / interactive)
get_script_dir <- function() {
  args <- commandArgs(FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m)) return(dirname(normalizePath(sub("^--file=", "", m[1]))))
  if (!is.null(sys.frames()[[1]]$ofile)) return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
  return(normalizePath(getwd()))
}
setwd(get_script_dir())

source("pandemic_priors.R")
suppressMessages(library(readxl))

# ============================ USER SETTINGS ==============================
nAR <- 12            # VAR lags
covid_periods <- 6   # pandemic dummy months, from Mar/2020 (h)
nimp <- 12           # IRF / forecast horizon (months)
rps <- 10000         # posterior draws
shocks <- 1          # recursive (Cholesky) shocks to compute. Either a count
                     # (1 = first shock, EBP) or a 0/1 selector over the variables,
                     # e.g. c(1,0,0,1,0,0,0,0) = shocks 1 & 4. Only selected shocks
                     # are computed (no extra cost).
constant <- 1        # include intercept
delta <- 1           # prior mean of own first lag (1 = levels/random walk,
                     # 0 = white noise/differences). A SCALAR applied to all
                     # variables, OR a length-nvar vector to set it per variable,
                     # e.g. c(1,1,0,1,1,1,1,0) to center rates on white noise and
                     # the rest on a random walk. Enters element-wise in the
                     # Minnesota block (sigma*delta) and sum-of-coefficients (delta*mu).
epsilon <- 0.001     # diffuse prior on the constant
seed <- 1            # rng seed for reproducible bands (NULL to skip)
screen <- TRUE       # stationarity screen: reject draws with an explosive companion
                     # root (max|eig| >= 1.01).
                     #   TRUE  -> default
                     #   FALSE -> keep every draw; faster loop, but retained
                     #            explosive draws fatten the tails.

# --- how the pandemic-dummy shrinkage phi is chosen ---
phi_mode <- "perperiod"  # "single"    : one phi, chosen on phi_grid
                         # "perperiod" : phi_1..phi_h, chosen by the optimiser
                         # <scalar>    : a fixed phi (e.g. 0.1; ~0 = uninformative)
lambda_mode <- 0.2       # Minnesota tightness: a scalar (0.2 default) or "optimal"
tau_mode <- "10lambda"   # sum-of-coefficients tightness: a scalar, "10lambda"
                         # (default, tau = 10*lambda), or "optimal". Set tau_mode=0
                         # to drop the sum-of-coefficients prior entirely.
coper_mode <- 0          # co-persistence / dummy-initial-observation prior
                         # (Sims-Zha 1998): a single extra dummy built from the
                         # average of the initial nAR observations, favouring a
                         # shared stochastic trend. 0 = OFF (paper default);
                         # a positive scalar sets a fixed tightness (smaller =
                         # tighter); or "optimal" to select it by marginal
                         # likelihood alongside phi (and any optimal lambda/tau).
# Selection: for phi_mode "single" every free hyperparameter is grid-searched (same
# grid as phi); for "perperiod" free lambda/tau/coper join the optimiser. When
# lambda is "optimal" and tau is "10lambda", tau tracks 10*lambda automatically.
# "optimal" coper searches a positive tightness (never 0/off) -- leave coper_mode=0
# to keep the co-persistence prior switched off.

data_file <- "./data/Data_20260524.xlsx"
log_vector <- c(0, 1, 0, 1, 1, 1, 1, 0)
Yname <- c("EBP", "S&P 500", "Shadow Rate", "PCE", "PCE Price Index",
           "Employment", "Ind. Production", "Unemp. Rate")
# Sample and pandemic window, set by date. Only rows in [sample_start, sample_end]
# are used, so the data file may extend past sample_end.
sample_start   <- as.Date("1975-01-01")
sample_end     <- as.Date("2025-12-01")
pandemic_start <- as.Date("2020-03-01")             # first pandemic dummy month
phi_grid <- c(0.001, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30,
              0.35, 0.40, 0.45, 0.50, 0.75, 1, 2, 5)

# ============================ BUILD Y, X ================================
raw <- read_excel(data_file)
# first column = monthly timestamps; take the calendar year/month in UTC (the
# tz readxl stores them in) and normalize to the first of the month.
lt <- as.POSIXlt(raw[[1]], tz = "UTC")
dates <- as.Date(sprintf("%04d-%02d-01", lt$year + 1900L, lt$mon + 1L))
keep <- which(dates >= sample_start & dates <= sample_end)
datevec <- dates[keep]
data <- as.matrix(raw[keep, -1]); data <- matrix(as.numeric(data), length(keep))
Yraw <- data
for (e in seq_along(log_vector)) if (log_vector[e] == 1) Yraw[, e] <- log(Yraw[, e]) * 100
Traw <- nrow(Yraw); nvar <- ncol(Yraw)

Ylag <- mlag2(Yraw, nAR)
if (constant == 1) X <- cbind(Ylag[(nAR + 1):Traw, ], 1) else X <- Ylag[(nAR + 1):Traw, ]
Y <- Yraw[(nAR + 1):Traw, ]

# pandemic dummy rows: locate the pandemic-start month in the estimation sample
covid_ind <- which(datevec == pandemic_start) - nAR
X <- cbind(X, matrix(0, nrow(X), covid_periods))
X[covid_ind:(covid_ind + covid_periods - 1), (ncol(X) - covid_periods + 1):ncol(X)] <-
  diag(covid_periods)

# ==================== SELECT HYPERPARAMETERS ============================
t0 <- Sys.time()
sel <- pp_select(phi_mode, lambda_mode, tau_mode, X, Y, Yraw, nAR, constant,
                 delta, epsilon, covid_periods, phi_grid, coper_mode)
t_phi <- as.numeric(Sys.time() - t0, units = "secs")

# ============================ ESTIMATE =================================
pp <- pandemicpriors(X, Y, Yraw, nAR, constant, delta, sel$lambda, sel$tau,
                     epsilon, sel$phi, covid_periods, sel$coper)
t0 <- Sys.time()
S <- pp_draws(X, Y, pp$xd, pp$yd, nAR, covid_periods, nimp, shocks, rps, seed, screen)
t_draw <- as.numeric(Sys.time() - t0, units = "secs")

cat(sprintf("phi-search %.2fs | draws+fcst+IRF %.2fs | screen %s (discarded %d)\n",
            t_phi, t_draw, ifelse(screen, "ON", "OFF"), S$discarded))
cat(sprintf("A_post %dx%d, IRF %s, forecast %s\n",
            nrow(S$A_post), ncol(S$A_post),
            paste(dim(S$irf), collapse = "x"), paste(dim(S$fcst), collapse = "x")))

# ============================ FIGURES ==================================
bands <- c(50, 16, 84)
p_lines <- nvar %/% 3; p_cols <- ceiling(nvar / p_lines); xax <- 1:nimp

# median + 68% band panel (grey ribbon between the 16th and 84th pct, red median)
band_panel <- function(q, ttl, zeroline = FALSE) {
  plot(xax, q[1, ], type = "n", main = ttl, xlab = "months", ylab = "", ylim = range(q))
  polygon(c(xax, rev(xax)), c(q[2, ], rev(q[3, ])), col = "grey85", border = NA)
  if (zeroline) abline(h = 0, lty = 3)
  lines(xax, q[1, ], col = "red", lwd = 2)
}

# ---- IRF figures (one per requested shock) ----
for (r in S$shocks) {
  pdf(sprintf("irf_shock%d.pdf", r), width = 11, height = 4.8)
  par(mfrow = c(p_lines, p_cols), oma = c(0, 0, 2, 0))
  for (uu in 1:nvar) band_panel(pp_prctile(S$irf[r, , , uu], bands), Yname[uu], zeroline = TRUE)
  mtext(sprintf("%s shock", Yname[r]), outer = TRUE, cex = 1.1)
  invisible(dev.off())
}

# ---- forecast fan charts ----
pdf("forecast.pdf", width = 11, height = 4.8)
par(mfrow = c(p_lines, p_cols), oma = c(0, 0, 2, 0))
for (uu in 1:nvar) band_panel(pp_prctile(S$fcst[, uu, ], bands), Yname[uu])
mtext("Forecast", outer = TRUE, cex = 1.1)
invisible(dev.off())
cat("wrote irf_shock*.pdf and forecast.pdf\n")
