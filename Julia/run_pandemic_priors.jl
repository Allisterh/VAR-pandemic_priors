# Pandemic Priors -- Julia driver (external release)
#
# Mirror of the MATLAB script run_pandemic_priors.m: builds the eight-variable
# monthly BVAR, selects phi, draws the posterior, and produces the EBP-shock
# impulse responses and forecast fan charts.
#
# Run:  julia run_pandemic_priors.jl
# Packages: Optim (used inside pp_select), XLSX (read the data), Plots (figures).
#   ] add Optim XLSX Plots
#
# The core (PandemicPriors.jl) itself uses only stdlib + SpecialFunctions + Optim.

cd(@__DIR__)                                                      # run from this script's folder
isdefined(Main, :PandemicPriors) || include("PandemicPriors.jl")  # include once (REPL-safe)
using .PandemicPriors
using XLSX, Statistics, LinearAlgebra, Dates

# ============================ USER SETTINGS ==============================
nAR = 12            # VAR lags
covid_periods = 6   # pandemic dummy months, from Mar/2020 (h)
nimp = 12           # IRF / forecast horizon (months)
rps = 10000         # posterior draws
shocks = 1          # recursive (Cholesky) shocks to compute. Either a count
                    # (1 = first shock, EBP) or a 0/1 selector over the variables,
                    # e.g. [1,0,0,1,0,0,0,0] = shocks 1 & 4. Only selected shocks
                    # are computed (no extra cost).
constant = 1        # include intercept
δ = 1.0             # prior mean of own first lag (1 = levels/random walk,
                    # 0 = white noise/differences). A SCALAR applied to all
                    # variables, OR a length-nvar vector to set it per variable,
                    # e.g. [1,1,0,1,1,1,1,0] to center rates on white noise and the
                    # rest on a random walk. Enters element-wise in the Minnesota
                    # block (σ.*δ) and the sum-of-coefficients block (δ.*μ).
ϵ = 0.001           # diffuse prior on the constant
seed = 1            # rng seed for reproducible bands (nothing to skip)
screen = true       # stationarity screen: reject draws with an explosive companion
                    # root (max|eig| >= 1.01).
                    #   true  -> default
                    #   false -> keep every draw; faster loop, but retained
                    #            explosive draws fatten the tails.

# --- how the pandemic-dummy shrinkage ϕ is chosen ---
ϕ_mode = "perperiod"  # "single"    : one ϕ, chosen on ϕ_grid
                      # "perperiod" : ϕ_1..ϕ_h, chosen by the optimiser
                      # <scalar>    : a fixed ϕ (e.g. 0.1; ~0 = uninformative)
λ_mode = 0.2          # Minnesota tightness: a scalar (0.2 default) or "optimal"
τ_mode = "10lambda"   # sum-of-coefficients tightness: a scalar, "10lambda"
                      # (default, τ = 10λ), or "optimal". Set τ_mode=0 to drop the
                      # sum-of-coefficients prior entirely.
γc_mode = 0.0          # co-persistence / dummy-initial-observation prior
                      # (Sims-Zha 1998): a single extra dummy built from the average
                      # of the initial nAR observations, favouring a shared
                      # stochastic trend. 0 = OFF (paper default); a positive scalar
                      # sets a fixed tightness (smaller = tighter); or "optimal" to
                      # select it by marginal likelihood alongside ϕ (and any
                      # optimal λ/τ).
# Selection: for ϕ_mode "single" every free hyperparameter is grid-searched (same
# grid as ϕ); for "perperiod" free λ/τ/γc join the optimiser. When λ is "optimal"
# and τ is "10lambda", τ tracks 10λ automatically. "optimal" γc searches a positive
# tightness (never 0/off) -- leave γc_mode=0 to keep the co-persistence prior off.

data_file = joinpath("data", "Data_20260524.xlsx")
log_vector = [0, 1, 0, 1, 1, 1, 1, 0]
Yname = ["EBP", "S&P 500", "Shadow Rate", "PCE", "PCE Price Index",
         "Employment", "Ind. Production", "Unemp. Rate"]
# Sample and pandemic window, set by date. Only rows in [sample_start, sample_end]
# are used, so the data file may extend past sample_end.
sample_start   = Date(1975, 1, 1)
sample_end     = Date(2025, 12, 1)
pandemic_start = Date(2020, 3, 1)          # first pandemic dummy month
ϕ_grid = [0.001, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30,
          0.35, 0.40, 0.45, 0.50, 0.75, 1, 2, 5]

# ============================ BUILD Y, X ================================
xf = XLSX.readxlsx(data_file)
sheet = xf[XLSX.sheetnames(xf)[1]]
raw = sheet[:]                              # includes header row
dates = Date.(raw[2:end, 1])                # first column = monthly timestamps
keep  = findall(d -> sample_start <= d <= sample_end, dates)
datevec = dates[keep]
data = Float64.(raw[2:end, 2:end])[keep, :] # drop header + date column, keep sample
Yraw = copy(data)
for e in 1:length(log_vector)
    if log_vector[e] == 1
        Yraw[:, e] = log.(Yraw[:, e]) .* 100
    end
end
Traw, nvar = size(Yraw)

Ylag = mlag2(Yraw, nAR)
X = constant == 1 ? hcat(Ylag[nAR+1:Traw, :], ones(Traw - nAR)) : Ylag[nAR+1:Traw, :]
Y = Yraw[nAR+1:Traw, :]

# pandemic dummy rows: locate the pandemic-start month in the estimation sample
covid_ind = findfirst(==(pandemic_start), datevec) - nAR
X = hcat(X, zeros(size(X, 1), covid_periods))
X[covid_ind:covid_ind+covid_periods-1, end-covid_periods+1:end] = Matrix{Float64}(I, covid_periods, covid_periods)

# ==================== SELECT HYPERPARAMETERS ============================
t0 = time()
ϕ_use, λ, τ, γc = pp_select(ϕ_mode, λ_mode, τ_mode, X, Y, Yraw,
                           nAR, constant, δ, ϵ, covid_periods, ϕ_grid, γc_mode)
t_phi = time() - t0

# ============================ ESTIMATE =================================
_, _, xd, yd = pandemicpriors(X, Y, Yraw, nAR, constant, δ, λ, τ,
                              ϵ, ϕ_use, covid_periods, γc)
t0 = time()
S = pp_draws(X, Y, xd, yd, nAR, covid_periods, nimp, shocks, rps; seed=seed, screen=screen)
t_draw = time() - t0

println("phi-search $(round(t_phi,digits=2))s | draws+fcst+IRF $(round(t_draw,digits=2))s | screen $(screen ? "ON" : "OFF") (discarded $(S[:discarded]))")
println("A_post $(size(S[:A_post])), IRF $(size(S[:irf])), forecast $(size(S[:fcst]))")

# ============================ FIGURES ==================================
using Plots
using Plots.PlotMeasures: mm
bands = [50, 16, 84]; xax = collect(1:nimp)
grey = RGB(0.70, 0.70, 0.70); red = RGB(0.85, 0.11, 0.11)
p_lines = nvar ÷ 3; p_cols = ceil(Int, nvar / p_lines)

# median + 68% band panel (grey ribbon between the 16th and 84th pct, red median)
function band_panel!(plt, idx, q, ttl; zeroline=false)
    plot!(plt[idx], xax, q[1, :], ribbon=(q[1, :] .- q[2, :], q[3, :] .- q[1, :]),
          fillalpha=1.0, fillcolor=grey, linecolor=red, lw=2,
          title=ttl, titlefont=font(9), guidefont=font(7), tickfont=font(6),
          xlabel="months")
    zeroline && plot!(plt[idx], xax, zeros(nimp), linecolor=:black, ls=:dot, lw=1)
end

# generous canvas + inter-panel margins so titles / "months" labels are not clipped
figsize = (1100, 750)
mgn = 4mm

# ---- IRF figures (one per requested shock) ----
for r in S[:shocks]
    local plt = plot(layout=(p_lines, p_cols), size=figsize, legend=false,
                     plot_title="$(Yname[r]) shock", left_margin=mgn,
                     bottom_margin=mgn, top_margin=mgn)
    for uu in 1:nvar
        q = pp_prctile(S[:irf][r, :, :, uu], bands)     # (3 x nimp)
        band_panel!(plt, uu, q, Yname[uu]; zeroline=true)
    end
    savefig(plt, "irf_shock$(r).pdf")
end

# ---- forecast fan charts ----
plt = plot(layout=(p_lines, p_cols), size=figsize, legend=false, plot_title="Forecast",
           left_margin=mgn, bottom_margin=mgn, top_margin=mgn)
for uu in 1:nvar
    q = pp_prctile(S[:fcst][:, uu, :], bands)
    band_panel!(plt, uu, q, Yname[uu])
end
savefig(plt, "forecast.pdf")
println("wrote irf_shock*.pdf and forecast.pdf")
