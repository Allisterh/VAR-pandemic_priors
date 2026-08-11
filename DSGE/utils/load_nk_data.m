function D = load_nk_data(infl, rate, sample)
% LOAD_NK_DATA  Turn raw data_nk.csv into the NK model's three observables.
%
%   D = LOAD_NK_DATA(infl, rate, sample)
%
%   Builds [x pi i] (quarterly) from the merged FRED / Wu-Xia file:
%       x   output gap    = 100*(GDPC1 - GDPPOT)/GDPPOT            [percent]
%       pi  inflation     = 400*log(P_t/P_{t-1})                  [annualized %]
%       i   policy rate   = chosen nominal rate                   [annualized %]
%   Inflation is annualized (x400) so pi and i share the same annualized units,
%   which the Taylor rule i = phi_pi*pi + phi_x*x + v_m requires to be coherent.
%
%   All three series are demeaned over the estimation sample, because the model
%   lives in deviations from a zero steady state. The means
%   removed are returned in D.mu for reference.
%
%   INPUTS (all optional)
%     infl   : 'GDPDEF' (default) | 'PCEPILFE'      -- price index for inflation
%     rate   : 'SHADOW_SPLICED' (default) | 'FEDFUNDS'  -- policy-rate column
%     sample : [t0 t1] datetimes, inclusive (default [1985Q1 .. last full row])
%
%   OUTPUT struct D
%     .y        T-by-3 observables [x pi i], demeaned
%     .y_raw    T-by-3 before demeaning
%     .mu       1-by-3 sample means removed
%     .dates    T-by-1 datetime (quarter start)
%     .pandemic 1-by-3 row indices of 2020Q1,Q2,Q3 within the sample
%     .names    {'x','pi','i'}
%     .infl,.rate  the choices used
%
%   See also BUILD_DATA_NK, RUN_DSGE_USDATA.

if nargin < 1 || isempty(infl),   infl = 'GDPDEF';         end
if nargin < 2 || isempty(rate),   rate = 'SHADOW_SPLICED'; end

here = fileparts(mfilename('fullpath'));                 % .../DSGE/utils
T = readtable(fullfile(here,'..','data','data_nk.csv'),'TextType','string');
d = datetime(T.date);

% --- raw building blocks --------------------------------------------------
gap   = 100*(T.GDPC1 - T.GDPPOT)./T.GDPPOT;            % output gap, percent
P     = T.(infl);
infl_q= [NaN; 400*diff(log(P))];                       % annualized inflation
irate = T.(rate);

% drop rows with any NaN in the three needed series (e.g. first diff, trailing)
core  = [gap, infl_q, irate];
ok    = all(isfinite(core),2);

if nargin < 3 || isempty(sample)
    t0 = datetime(1985,1,1);
    t1 = max(d(ok));
else
    t0 = sample(1); t1 = sample(2);
end
keep = ok & d>=t0 & d<=t1;

dates = d(keep);
y_raw = core(keep,:);

% --- demean (model = deviations from zero steady state) -------------------
mu   = mean(y_raw,1);
y    = y_raw - mu;

% --- locate pandemic quarters --------------------------------------------
pand = find( dates>=datetime(2020,1,1) & dates<=datetime(2020,7,1) ).';
assert(numel(pand)==3, 'expected 3 pandemic quarters in sample; found %d', numel(pand));

D = struct('y',y,'y_raw',y_raw,'mu',mu,'dates',dates,'pandemic',pand, ...
           'names',{{'x','pi','i'}},'infl',infl,'rate',rate);

fprintf('load_nk_data: infl=%s rate=%s  sample %s..%s  (T=%d)\n', ...
        infl, rate, datestr(dates(1),'yyyyQQ'), datestr(dates(end),'yyyyQQ'), numel(dates));
fprintf('  means removed  x=%.2f  pi=%.2f  i=%.2f\n', mu(1), mu(2), mu(3));
end
