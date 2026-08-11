function s = nk_dispname(tag)
% NK_DISPNAME  Human-readable display name for a data-series mnemonic.
%   Maps raw series mnemonics to display text for use in figures.
%   Accepts a single mnemonic (char/string) or a combo like 'PCEPILFE_FEDFUNDS'
%   and returns 'core PCE inflation (ex.\ food \& energy) inflation, federal funds
%   rate'-style text (LaTeX-safe: escapes &).
switch upper(char(tag))
    case 'GDPDEF',         s = 'GDP deflator inflation';
    case 'PCEPILFE',       s = 'core PCE inflation (ex.\ food \& energy)';
    case 'FEDFUNDS',       s = 'federal funds rate';
    case 'SHADOW_SPLICED', s = 'Wu-Xia shadow rate';
    case 'CPIAUCSL',       s = 'CPI inflation';
    otherwise
        % combo "<infl>_<rate>": split on the first underscore-separated infl token
        parts = split(string(tag),'_');
        known = {'GDPDEF','PCEPILFE','CPIAUCSL'};
        if numel(parts)>=2 && any(strcmpi(char(parts(1)),known))
            infl = nk_dispname(char(parts(1)));
            rate = nk_dispname(char(strjoin(parts(2:end),'_')));
            s = sprintf('%s, %s', infl, rate);
        else
            s = char(tag);   % fallback: unknown tag, return as-is
        end
end
end
