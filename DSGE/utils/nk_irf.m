function irf = nk_irf(sol, shockname, H, shocksize)
% NK_IRF  Impulse responses of [x; pi; i] to a structural shock.
%
%   irf = NK_IRF(sol, shockname, H, shocksize) returns an (H+1)-by-3 matrix of
%   responses (rows = horizons 0..H, cols = [x pi i]) to a one-off innovation in
%   SHOCKNAME of size SHOCKSIZE (default: one standard deviation of that shock).
%
%   sol       : output of nk_solve
%   shockname : 'eps_a' | 'eps_s' | 'eps_m'   ('eps_m' = monetary)
%   H         : horizon in quarters (default 20)
%   shocksize : innovation size (default: 1 s.d. of the chosen shock)
%
%   The state follows s_t = F*s_{t-1} + eps_t with G = I, so the impulse in the
%   innovation propagates as s_h = F^h * e_k * shocksize and the observables are
%   Z*s_h, for a one-standard-deviation shock.

if nargin < 3 || isempty(H),         H = 20;                       end
k = find(strcmp(sol.shocknames, shockname));
if isempty(k), error('nk_irf:badShock','Unknown shock "%s".', shockname); end
if nargin < 4 || isempty(shocksize), shocksize = sqrt(sol.Q(k,k)); end

e = zeros(3,1);  e(k) = shocksize;      % innovation vector
irf = zeros(H+1, 3);
s   = e;                                % s_0 = eps_0 (impact)
for h = 0:H
    irf(h+1,:) = (sol.Z * s).';
    s = sol.F * s;                      % propagate: s_{h+1} = F*s_h
end
end
