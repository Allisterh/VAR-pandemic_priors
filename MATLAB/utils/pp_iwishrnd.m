function Sigma = pp_iwishrnd(Tau,df,C)
% PP_IWISHRND  Inverse-Wishart draw, toolbox-free (no Statistics Toolbox).
%
%   Sigma = pp_iwishrnd(Tau,df)      draws Sigma ~ IW(Tau,df)
%   Sigma = pp_iwishrnd(Tau,df,C)    reuses C = chol(Tau,'lower') (prime once)
%
%   Same parameterization as MATLAB's iwishrnd: Tau is the scale matrix, df the
%   degrees of freedom, and inv(Sigma) ~ Wishart(df, inv(Tau)); hence
%   E[Sigma] = Tau/(df-n-1). Drop-in replacement for iwishrnd(Tau,df[,DI]).
%
%   Method (no toolbox): Bartlett decomposition. With Tau = C*C' (C lower) and A
%   the lower-triangular Bartlett factor of a Wishart(df,I) draw
%       A(i,i) = sqrt(chi2_{df-i+1}),   A(i,j)~N(0,1) for i>j,   0 above,
%   an IW(Tau,df) draw is Sigma = K*K' with K = C * inv(A)'. The chi-square
%   variates come from a self-contained Marsaglia-Tsang gamma sampler (randn/rand
%   only): chi2_k = 2*gamma(k/2,1).
%
%   Danilo Cascaldi-Garcia

n = size(Tau,1);
if nargin<3 || isempty(C)
    C = chol(Tau,'lower');            % Tau = C*C'
end

% Bartlett lower-triangular factor A of a Wishart(df,I) sample
A = zeros(n);
for i=1:n
    A(i,i) = sqrt(2*randgamma((df-i+1)/2));   % chi2_{df-i+1}
end
if n>1
    L = tril(true(n),-1);
    A(L) = randn(sum(L(:)),1);                % strictly-lower ~ N(0,1)
end

% Sigma = C * inv(A)' * (C * inv(A)')'   (IW(Tau,df); no explicit matrix inverse)
K     = C / A';                                % = C * inv(A') , solved by back-sub
Sigma = K*K';
Sigma = (Sigma+Sigma')/2;                      % numerical symmetry guard
end

% ------------------------------------------------------------------------
function g = randgamma(a)
% Gamma(shape=a, scale=1) via Marsaglia & Tsang (2000). randn/rand only.
if a < 1                                       % boost low shape into a>=1
    g = randgamma(a+1) * rand^(1/a);
    return
end
d = a - 1/3;
c = 1/sqrt(9*d);
while true
    x = randn;
    v = (1 + c*x)^3;
    if v <= 0, continue; end
    u = rand;
    if log(u) < 0.5*x^2 + d - d*v + d*log(v)
        g = d*v;
        return
    end
end
end
