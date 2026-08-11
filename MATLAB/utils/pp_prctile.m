function q = pp_prctile(X,p,dim)
% PP_PRCTILE  Percentiles, toolbox-free (no Statistics Toolbox).
%
%   q = pp_prctile(X,p)      percentiles p (in 0..100) down the first non-
%                            singleton dimension.
%   q = pp_prctile(X,p,dim)  along dimension dim.
%
%   Matches MATLAB's prctile default convention: sample values sit at
%   plotting positions 100*(i-0.5)/N, linear interpolation in between, and
%   clamped to the min/max outside that range. Drop-in for prctile(X,p,dim).
%
%   Danilo Cascaldi-Garcia

if nargin<3 || isempty(dim)
    dim = find(size(X)~=1,1);
    if isempty(dim); dim = 1; end
end

% Move the working dimension to the front, collapse the rest into columns.
X   = sort(X,dim);
perm = [dim, setdiff(1:max(ndims(X),dim),dim)];
Xp  = permute(X,perm);
sz  = size(Xp);
N   = sz(1);
Xp  = reshape(Xp,N,[]);

p = p(:);                            % column of requested percentiles
if N==1
    Q = repmat(Xp,numel(p),1);       % single observation: constant
else
    pos = 100*((1:N)-0.5)/N;         % plotting positions of the sorted data
    Q = zeros(numel(p),size(Xp,2));
    for c=1:size(Xp,2)
        Q(:,c) = interp1(pos,Xp(:,c),p,'linear');
        Q(p<=pos(1),c)  = Xp(1,c);   % clamp below first position
        Q(p>=pos(end),c)= Xp(end,c); % clamp above last position
    end
end

% Reshape back: percentile index replaces the working dimension.
outsz    = sz; outsz(1) = numel(p);
q = reshape(Q,outsz);
q = ipermute(q,perm);
end
