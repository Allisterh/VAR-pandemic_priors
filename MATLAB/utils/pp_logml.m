function py = pp_logml(X,Y,Yraw,nAR,constant,delta,lambda,tau,epsilon,phi,covid_temp,coper)
% PP_LOGML  Fast log marginal likelihood of the Pandemic-Priors conjugate BVAR.
%
%   The log marginal likelihood is computed from Cholesky factors of the k-by-k
%   and n-by-n cross-product matrices rather than a T-by-T det/inv path, which
%   keeps the phi search fast.
%
%   Identities used:
%     det(I_T + X*inv(xx0)*X') = det(XXst)/det(xx0)          [matrix-det lemma]
%     QQ + (Y-X*b0)'*PP*(Y-X*b0) = SSE_post = Yst'Yst - A_post'*XXst*A_post
%   where xx0=xd'*xd, XXst=xd'*xd+X'*X, A_post=XXst\(xd'*yd+X'*Y).
%
%   Danilo Cascaldi-Garcia

% Prior via the same dummy-observation builder (coper passed explicitly)
[~, ~, xd, yd] = pandemicpriors(X,Y,Yraw,nAR,constant,delta,lambda,tau,epsilon,phi,covid_temp,coper);

T = size(X,1);
n = size(Y,2);
v0 = n + 2;
v1 = v0 + T;

% Prior cross-products (k-by-k) and prior scale (n-by-n)
xx0    = xd'*xd;                 % prior precision   (k x k)
Cxx0   = chol(xx0,'lower');      % logdet(xx0) = 2*sum(log(diag(Cxx0)))
b0     = Cxx0'\(Cxx0\(xd'*yd));  % prior coeff mean  = xx0 \ (xd'*yd)
e0     = yd - xd*b0;
sigma0 = e0'*e0;                 % prior scale QQ    (n x n)

% Posterior cross-products
XXst = xx0 + X'*X;               % = Xst'*Xst        (k x k)
CXX  = chol(XXst,'lower');
XYst = xd'*yd + X'*Y;            % = Xst'*Yst
Apo  = CXX'\(CXX\XYst);          % posterior mean    = XXst \ XYst

% Posterior scale S1 = Yst'Yst - Apo'*XXst*Apo  (= posterior IW scale SSE_post)
YYst = yd'*yd + Y'*Y;
S1   = YYst - XYst'*Apo;         % symmetric; XYst'*Apo == Apo'*XXst*Apo
S1   = (S1 + S1')/2;             % numerical symmetry guard
CS1  = chol(S1,'lower');
Cs0  = chol(sigma0,'lower');

% log determinants from Cholesky diagonals (small matrices only)
ldet_xx0   = 2*sum(log(diag(Cxx0)));
ldet_XXst  = 2*sum(log(diag(CXX)));
ldet_sigma0= 2*sum(log(diag(Cs0)));
ldet_S1    = 2*sum(log(diag(CS1)));

r1 = logMvGamma(v0/2,n);
r2 = logMvGamma(v1/2,n);

% (n/2)*log det(PP) = (n/2)*(ldet_xx0 - ldet_XXst); (v0/2)*ldet_sigma0; -(v1/2)*ldet_S1
py = -(T*n/2)*log(pi) ...
     + (n/2)*(ldet_xx0 - ldet_XXst) ...
     + (v0/2)*ldet_sigma0 ...
     + (r2 - r1) ...
     - (v1/2)*ldet_S1;
end
