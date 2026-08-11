# Pandemic Priors -- Julia implementation (external release)
#
# Port of the MATLAB replication code for Cascaldi-Garcia, D., "Pandemic Priors."
# Standard-library only (LinearAlgebra, Statistics, Random, SpecialFunctions),
# plus Optim for the derivative-free hyperparameter search.
#
# Julia lets the code use the paper's own Greek notation, so the mathematical
# parameters read as in the manuscript:
#     λ  overall (Minnesota) tightness            τ  sum-of-coefficients tightness
#     ϕ  pandemic-dummy shrinkage (scalar or ϕ_1..ϕ_h)
#     δ  prior mean of the own first lag          ϵ  diffuse intercept prior
#     σ  Minnesota scale factors                  μ  variable means
#     γc  co-persistence (dummy-initial-obs) tightness   Σ  residual covariance
#
# Deterministic quantities (selected ϕ, A_post, marginal likelihood) match the
# MATLAB/Python/R output to numerical tolerance. Posterior bands match in
# distribution, not draw-for-draw (RNG stream differs across languages).
#
# Use of code for research purposes is permitted with proper reference.
# Danilo Cascaldi-Garcia.

module PandemicPriors

using LinearAlgebra, Statistics, Random
using SpecialFunctions: loggamma
using Optim  # Nelder-Mead for pp_select (perperiod)

export mlag2, pandemicpriors, pp_logml, pp_select, pp_draws, pp_iwishrnd, pp_prctile

# ---------------------------------------------------------------------------
function mlag2(X::AbstractMatrix, p::Int)
    Traw, N = size(X)
    Xlag = zeros(Traw, N * p)
    for ii in 1:p
        Xlag[p+1:Traw, N*(ii-1)+1:N*ii] = X[p+1-ii:Traw-ii, :]
    end
    return Xlag
end

# ---------------------------------------------------------------------------
"log multivariate gamma Γ_d(x)"
function logmvgamma(x::Real, d::Int)
    j = (1 .- (1:d)) ./ 2
    return d * (d - 1) / 4 * log(pi) + sum(loggamma.(x .+ j))
end

# ---------------------------------------------------------------------------
"""
    pandemicpriors(X, Y, Yraw, nAR, constant, δ, λ, τ, ϵ, ϕ, covid_periods, γc=0)

Build the Pandemic-Priors dummy-observation matrices. Returns (Xst, Yst, xd, yd).
`ϕ` may be a scalar or a length-h vector; `δ` a scalar or length-n; `γc = 0`
switches off the co-persistence (Sims-Zha) prior.
"""
function pandemicpriors(X, Y, Yraw, nAR, constant, δ, λ, τ, ϵ, ϕ, covid_periods, γc=0.0)
    X = Matrix{Float64}(X); Y = Matrix{Float64}(Y); Yraw = Matrix{Float64}(Yraw)
    T = size(X, 1); Traw = T + nAR; n = size(Y, 2)
    μ = vec(mean(Y, dims=1))                       # BGR full-sample mean

    ϕ = length(ϕ) == 1 ? fill(Float64(ϕ[1]), covid_periods) : Float64.(vec(ϕ))
    δ = length(δ) == 1 ? fill(Float64(δ[1]), n) : Float64.(vec(δ))

    # univariate AR(nAR) per variable -> σ (Minnesota scale factors)
    ResidAR = fill(NaN, Traw - nAR, n)
    for i in 1:n
        YlagAR = fill(NaN, Traw, nAR)
        for ii in 1:nAR
            YlagAR[nAR+1:Traw, ii] = Yraw[nAR+1-ii:Traw-ii, i]
        end
        X_AR = hcat(YlagAR[nAR+1:Traw, :], ones(Traw - nAR))
        Y_AR = Yraw[nAR+1:Traw, i]
        β_AR = X_AR \ Y_AR
        ResidAR[:, i] = Y_AR - X_AR * β_AR
    end
    Σ_AR = (ResidAR' * ResidAR) ./ size(ResidAR, 1)
    σ = sqrt.(diag(Σ_AR))

    yd = Array{Float64}(undef, 0, n)
    ncolx = n * nAR + covid_periods + (constant == 1 ? 1 : 0)
    xd = Array{Float64}(undef, 0, ncolx)

    if λ > 0
        Jp = Matrix(Diagonal(Float64.(1:nAR)))
        bb = Matrix(Diagonal(σ .* δ ./ λ))
        dd = Matrix(Diagonal(σ))
        ff = Matrix(Diagonal(σ ./ λ))
        kjf = kron(Jp, ff)
        if constant == 1
            yd1 = vcat(bb, zeros(n * (nAR - 1), n), dd, zeros(covid_periods + 1, n))
            top = hcat(kjf, zeros(n * nAR, covid_periods + 1))
            mid = zeros(n, n * nAR + covid_periods + 1)
            crow = hcat(zeros(1, n * nAR), reshape([ϵ], 1, 1), zeros(1, covid_periods))
            drows = hcat(zeros(covid_periods, n * nAR + 1), Matrix(Diagonal(ϕ)))
            xd1 = vcat(top, mid, crow, drows)
        else
            yd1 = vcat(bb, zeros(n * (nAR - 1), n), dd, zeros(covid_periods, n))
            top = hcat(kjf, zeros(n * nAR, covid_periods))
            mid = zeros(n, n * nAR + covid_periods)
            drows = hcat(zeros(covid_periods, n * nAR), Matrix(Diagonal(ϕ)))
            xd1 = vcat(top, mid, drows)
        end
        yd = vcat(yd, yd1); xd = vcat(xd, xd1)

        if τ > 0
            bb2 = Matrix(Diagonal(δ .* μ ./ τ))
            kv = kron(ones(1, nAR), bb2)
            pad = covid_periods + (constant == 1 ? 1 : 0)
            xd2 = hcat(kv, zeros(n, pad))
            yd = vcat(yd, bb2); xd = vcat(xd, xd2)
        end
    end

    if γc > 0
        y0 = vec(mean(Yraw[1:nAR, :], dims=1))     # average of the initial conditions
        yd3 = reshape(y0 ./ γc, 1, n)
        ky = kron(ones(1, nAR), yd3)
        if constant == 1
            xd3 = hcat(ky, reshape([1.0 / γc], 1, 1), zeros(1, covid_periods))
        else
            xd3 = hcat(ky, zeros(1, covid_periods))
        end
        yd = vcat(yd, yd3); xd = vcat(xd, xd3)
    end

    Yst = vcat(Y, yd); Xst = vcat(X, xd)
    return Xst, Yst, xd, yd
end

# ---------------------------------------------------------------------------
"Fast closed-form log marginal likelihood of the Pandemic-Priors conjugate BVAR."
function pp_logml(X, Y, Yraw, nAR, constant, δ, λ, τ, ϵ, ϕ, covid_periods, γc=0.0)
    X = Matrix{Float64}(X); Y = Matrix{Float64}(Y)
    _, _, xd, yd = pandemicpriors(X, Y, Yraw, nAR, constant, δ, λ, τ, ϵ, ϕ, covid_periods, γc)
    T = size(X, 1); n = size(Y, 2); v0 = n + 2; v1 = v0 + T

    xx0 = xd' * xd
    Cxx0 = cholesky(Symmetric(xx0)).L
    b0 = Cxx0' \ (Cxx0 \ (xd' * yd))
    e0 = yd - xd * b0
    Σ0 = e0' * e0

    XXst = xx0 + X' * X
    CXX = cholesky(Symmetric(XXst)).L
    XYst = xd' * yd + X' * Y
    Apo = CXX' \ (CXX \ XYst)

    YYst = yd' * yd + Y' * Y
    S1 = YYst - XYst' * Apo
    S1 = (S1 + S1') / 2
    CS1 = cholesky(Symmetric(S1)).L
    Cs0 = cholesky(Symmetric(Σ0)).L

    ldet_xx0 = 2 * sum(log.(diag(Cxx0)))
    ldet_XXst = 2 * sum(log.(diag(CXX)))
    ldet_Σ0 = 2 * sum(log.(diag(Cs0)))
    ldet_S1 = 2 * sum(log.(diag(CS1)))

    r1 = logmvgamma(v0 / 2, n); r2 = logmvgamma(v1 / 2, n)
    return -(T * n / 2) * log(pi) + (n / 2) * (ldet_xx0 - ldet_XXst) +
           (v0 / 2) * ldet_Σ0 + (r2 - r1) - (v1 / 2) * ldet_S1
end

# ---------------------------------------------------------------------------
"""
    pp_select(ϕ_mode, λ_mode, τ_mode, X, Y, Yraw, nAR, constant, δ, ϵ,
              covid_periods, ϕ_grid, γc_mode=0; verbose=true)

Choose (ϕ, λ, τ, γc) by maximizing the marginal likelihood. Returns (ϕ, λ, τ, γc).
`ϕ_mode` = "single" (grid), "perperiod" (optimiser), or a fixed scalar/vector;
`λ_mode`/`τ_mode`/`γc_mode` are scalars, "optimal", or (for τ) "10lambda".
"""
function pp_select(ϕ_mode, λ_mode, τ_mode, X, Y, Yraw, nAR, constant, δ, ϵ,
                   covid_periods, ϕ_grid, γc_mode=0.0; verbose=true)
    is_str(s) = s isa AbstractString
    λ_opt = is_str(λ_mode) && lowercase(λ_mode) == "optimal"
    τ_opt = is_str(τ_mode) && lowercase(τ_mode) == "optimal"
    τ_tie = is_str(τ_mode) && lowercase(τ_mode) in ("10lambda", "10*lambda")
    γc_opt = is_str(γc_mode) && lowercase(γc_mode) == "optimal"

    λ_fix = λ_opt ? 0.0 : Float64(λ_mode)
    τ_fix = (τ_opt || τ_tie) ? 0.0 : Float64(τ_mode)
    γc_fix = γc_opt ? 0.0 : Float64(γc_mode)

    lb, ub = minimum(ϕ_grid), maximum(ϕ_grid)
    g(z) = lb .+ (ub - lb) ./ (1 .+ exp.(-z))
    ginv(pv) = log((pv - lb) / (ub - pv))

    is_perperiod = is_str(ϕ_mode) && lowercase(ϕ_mode) == "perperiod"
    is_single = is_str(ϕ_mode) && lowercase(ϕ_mode) == "single"

    local ϕ_use, λ, τ, γc
    if is_perperiod
        h = covid_periods
        idx_λ = idx_τ = idx_γc = 0; npar = h
        if λ_opt; npar += 1; idx_λ = npar; end
        if τ_opt; npar += 1; idx_τ = npar; end
        if γc_opt; npar += 1; idx_γc = npar; end

        function negML(z)
            ϕ = g(z[1:h])
            λ_ = λ_opt ? g(z[idx_λ]) : λ_fix
            τ_ = τ_opt ? g(z[idx_τ]) : (τ_tie ? 10 * λ_ : τ_fix)
            γc_ = γc_opt ? g(z[idx_γc]) : γc_fix
            return -pp_logml(X, Y, Yraw, nAR, constant, δ, λ_, τ_, ϵ, ϕ,
                             covid_periods, γc_)
        end
        best = Inf; zbest = nothing
        for s0 in (0.3, 1.0, 1.7)
            z0 = fill(ginv(s0), npar)
            if idx_λ > 0; z0[idx_λ] = ginv(0.2); end
            if idx_τ > 0; z0[idx_τ] = ginv(min(2.0, ub)); end
            if idx_γc > 0; z0[idx_γc] = ginv(1.0); end
            # NelderMead convergence in Optim is governed by g_tol (spread of the
            # simplex vertex values), not x_tol/f_tol; loosen it so it stops early.
            res = optimize(negML, z0, NelderMead(),
                           Optim.Options(iterations=2000, g_tol=1e-6))
            if Optim.minimum(res) < best
                best = Optim.minimum(res); zbest = Optim.minimizer(res)
            end
        end
        ϕ_use = g(zbest[1:h])
        λ = λ_opt ? g(zbest[idx_λ]) : λ_fix
        τ = τ_opt ? g(zbest[idx_τ]) : (τ_tie ? 10 * λ : τ_fix)
        γc = γc_opt ? g(zbest[idx_γc]) : γc_fix
    else
        n_extra_opt = Int(λ_opt) + Int(τ_opt) + Int(γc_opt)
        if n_extra_opt >= 2
            @warn "Grid search with multiple 'optimal' hyperparameters multiplies the number of ML evaluations; consider ϕ_mode='perperiod'."
        end
        ϕ_cands = is_single ? collect(ϕ_grid) : [ϕ_mode]
        λ_vals = λ_opt ? ϕ_grid : [λ_fix]
        γc_vals = γc_opt ? ϕ_grid : [γc_fix]
        best = -Inf
        ϕ_use = ϕ_cands[1]; λ = λ_vals[1]; τ = NaN; γc = γc_vals[1]
        for λv in λ_vals
            τ_vals = τ_tie ? [10 * λv] : (τ_opt ? ϕ_grid : [τ_fix])
            for τv in τ_vals, γcv in γc_vals, ϕv in ϕ_cands
                ml = pp_logml(X, Y, Yraw, nAR, constant, δ, λv, τv, ϵ, ϕv,
                              covid_periods, γcv)
                if ml > best
                    best = ml; ϕ_use = ϕv; λ = λv; τ = τv; γc = γcv
                end
            end
        end
    end
    if verbose
        ϕstr = length(ϕ_use) == 1 ?
            string(round(Float64(ϕ_use[1]), digits=3)) :
            "[" * join([string(round(v, digits=3)) for v in ϕ_use], ", ") * "]"
        println("Selected ϕ = $ϕstr | λ = $(round(λ,digits=3)) | τ = $(round(τ,digits=3)) | γc = $(round(γc,digits=3))")
    end
    return ϕ_use, λ, τ, γc
end

# ---------------------------------------------------------------------------
"Gamma(shape=a, scale=1) via Marsaglia-Tsang."
function _randgamma(a, rng)
    if a < 1
        return _randgamma(a + 1, rng) * rand(rng)^(1 / a)
    end
    d = a - 1 / 3; c = 1 / sqrt(9 * d)
    while true
        x = randn(rng); v = (1 + c * x)^3
        v <= 0 && continue
        u = rand(rng)
        if log(u) < 0.5 * x^2 + d - d * v + d * log(v)
            return d * v
        end
    end
end

"Inverse-Wishart draw Σ ~ IW(Ψ, df) via Bartlett decomposition."
function pp_iwishrnd(Ψ, df, C, rng)
    n = size(Ψ, 1)
    A = zeros(n, n)
    for i in 1:n
        A[i, i] = sqrt(2 * _randgamma((df - (i - 1)) / 2, rng))
    end
    if n > 1
        for i in 2:n, j in 1:(i-1)
            A[i, j] = randn(rng)
        end
    end
    K = (A \ C')'          # C * inv(A')
    Σ = K * K'
    return (Σ + Σ') / 2
end

# ---------------------------------------------------------------------------
"Percentiles down columns, matching MATLAB's default plotting-position convention."
function pp_prctile(X::AbstractMatrix, p)
    N, m = size(X)
    if N == 1
        return repeat(X, length(p), 1)
    end
    pos = 100 .* ((1:N) .- 0.5) ./ N
    Q = zeros(length(p), m)
    for c in 1:m
        xs = sort(X[:, c])
        for (k, pk) in enumerate(p)
            if pk <= pos[1]
                Q[k, c] = xs[1]
            elseif pk >= pos[end]
                Q[k, c] = xs[end]
            else
                j = searchsortedlast(pos, pk)
                t = (pk - pos[j]) / (pos[j+1] - pos[j])
                Q[k, c] = xs[j] + t * (xs[j+1] - xs[j])
            end
        end
    end
    return Q
end

# ---------------------------------------------------------------------------
"""
    pp_draws(X, Y, xd, yd, nAR, covid_periods, nimp, shocks, rps; seed, screen)

Posterior draws + forecast + recursive (Cholesky) IRF. Returns a Dict with keys
:A_post, :SSE_post, :v1, :irf (n×rps×nimp×n), :fcst (rps×n×nimp), :C_auto,
:C_dummies, :discarded, :shocks.
"""
function pp_draws(X, Y, xd, yd, nAR, covid_periods, nimp, shocks, rps;
                  seed=nothing, screen=true)
    X = Matrix{Float64}(X); Y = Matrix{Float64}(Y)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    T, k = size(X); n = size(Y, 2); nvar = n
    sizeCF = nAR * nvar

    shock_idx = length(shocks) == 1 ? collect(1:shocks[1]) : findall(!=(0), vec(shocks))

    Xst = vcat(X, xd); Yst = vcat(Y, yd)
    XXst = Xst' * Xst
    C = cholesky(Symmetric(XXst)).L
    XYst = Xst' * Yst
    A_post = C' \ (C \ XYst)
    RESID = Yst - Xst * A_post
    SSE_post = RESID' * RESID; SSE_post = (SSE_post + SSE_post') / 2
    v1 = size(Xst, 1) + 2 - size(Xst, 2)
    CSSE = cholesky(Symmetric(SSE_post)).L

    irf = zeros(n, rps, nimp, n)
    fcst = zeros(rps, n, nimp)
    C_auto = zeros(rps, n, n)
    C_dummies = zeros(rps, 1 + covid_periods, n)
    discarded = 0

    Ylast = Y[T, :]
    Xtail = vcat(X[T, 1:nvar*(nAR-1)], X[T, nvar*nAR+1:k])

    Acomp = zeros(nvar * nAR, nvar * nAR)
    Acomp[nvar+1:nvar*nAR, 1:nvar*nAR-nvar] = Matrix{Float64}(I, nvar*nAR-nvar, nvar*nAR-nvar)

    for iii in 1:rps
        local Σrep, β
        while true
            Σrep = pp_iwishrnd(SSE_post, v1, CSSE, rng)
            Z = randn(rng, k, n)
            β = A_post + (C' \ Z) * cholesky(Symmetric(Σrep)).U
            Acomp[1:nvar, :] = β[1:nvar*nAR, :]'
            if !screen; break; end
            roots = sort(abs.(eigvals(Acomp)))
            if roots[sizeCF] < 1.01; break; else; discarded += 1; end
        end
        A0hat = Matrix(cholesky(Symmetric(Σrep)).L)

        C_auto[iii, :, :] = β[1:nvar, 1:nvar]
        C_dummies[iii, :, :] = β[nvar*nAR+1:end, :]

        Y_for = copy(Ylast); X_for = vcat(Ylast, Xtail)
        for ww in 1:nimp
            if ww > 1
                Y_for = vec(X_for' * β)
                X_for = vcat(Y_for, X_for[1:nvar*(nAR-1)], X_for[nvar*nAR+1:end])
            end
            fcst[iii, :, ww] = Y_for
        end

        for r in shock_idx
            imp = zeros(nimp, nvar)
            z = vcat(A0hat[:, r], zeros(nvar * nAR - nvar))
            imp[1, :] = z[1:nvar]
            for kk in 2:nimp
                z = Acomp * z
                imp[kk, :] = z[1:nvar]
            end
            if imp[1, r] < 0; imp = -imp; end
            irf[r, iii, :, :] = imp
        end
    end
    return Dict(:A_post => A_post, :SSE_post => SSE_post, :v1 => v1, :irf => irf,
                :fcst => fcst, :C_auto => C_auto, :C_dummies => C_dummies,
                :discarded => discarded, :shocks => shock_idx)
end

end # module
