% BUILD_DATA_NK  Assemble the single US-data dataset  data_nk.csv.
%
%   Merges the raw FRED quarterly series and the Atlanta-Fed Wu-Xia shadow rate
%   onto a common quarterly grid (quarter-start dates, FRED convention).
%
%   Raw levels/rates are kept untransformed (the NK detrending -- output gap,
%   inflation differencing, demeaning -- happens later in the data loader,
%   because the model is in deviations from a zero steady state).
%
%   Policy-rate construction (SHADOW_SPLICED):
%     - Wu-Xia is monthly and ends 2022-02. It is quarterly-averaged for every
%       quarter that is fully covered (all 3 months present) -> through 2021Q4.
%     - From the first not-fully-covered quarter onward (2022Q1+), the quarterly
%       effective federal funds rate (FEDFUNDS) is spliced in.
%     - This is exact: away from the zero lower bound the shadow rate equals the
%       policy rate, and Wu-Xia had already risen back through zero by Feb 2022,
%       so the splice introduces no level break.
%     Columns WUXIA_Q (raw quarterly Wu-Xia, NaN after 2021Q4) and FEDFUNDS are
%     both kept for transparency; SHADOW_SPLICED is the merged policy rate.
%
%   Output: data_nk.csv with columns
%     date, GDPC1, GDPPOT, GDPDEF, PCEPILFE, CPIAUCSL, FEDFUNDS, WUXIA_Q, SHADOW_SPLICED
%
%   Usage:  >> build_data_nk

clear; clc;
here = fileparts(mfilename('fullpath')); cd(here);

% ------------------------------------------------ common quarterly grid ---
grid = (datetime(1960,1,1):calquarters(1):datetime(2026,1,1)).';
G = table(grid,'VariableNames',{'date'});

% ------------------------------------------------ FRED quarterly series ---
fred = {'GDPC1','GDPPOT','GDPDEF','PCEPILFE','CPIAUCSL','FEDFUNDS'};
for k = 1:numel(fred)
    T = readtable([fred{k} '.csv'],'TextType','string');
    d = datetime(T.observation_date);               % 'yyyy-MM-dd'
    d = dateshift(d,'start','quarter');              % snap to quarter start
    v = T.(fred{k});
    tmp = table(d, v, 'VariableNames',{'date', fred{k}});
    tmp = tmp(~isnat(tmp.date),:);
    G = outerjoin(G, tmp, 'Keys','date','MergeKeys',true,'Type','left');
end

% ------------------------------------------------ Wu-Xia (monthly xlsx) ---
[num,~,raw] = xlsread('WuXiaShadowRate.xlsx','Data');
% raw: row1 header; col1 date (M/D/YYYY as text), col2 eff FFR, col3 Wu-Xia
n   = size(raw,1);
wd  = NaT(n,1);  ws = nan(n,1);
for r = 2:n
    ds = raw{r,1};
    if ischar(ds) || isstring(ds)
        wd(r) = datetime(ds,'InputFormat','M/d/yyyy');
    elseif isnumeric(ds) && ~isnan(ds)
        wd(r) = datetime(ds,'ConvertFrom','excel');
    end
    if isnumeric(raw{r,3}), ws(r) = raw{r,3}; end
end
ok = ~isnat(wd) & ~isnan(ws);
wd = wd(ok);  ws = ws(ok);
fprintf('Wu-Xia monthly sample: %s .. %s (%d obs)\n', ...
    datestr(min(wd),'yyyy-mm'), datestr(max(wd),'yyyy-mm'), numel(wd));

% quarterly average, only for fully covered quarters (3 months present)
qkey  = dateshift(wd,'start','quarter');
uq    = unique(qkey);
wuxiaQ = nan(numel(uq),1);  cnt = zeros(numel(uq),1);
for i = 1:numel(uq)
    sel = qkey==uq(i);
    cnt(i) = nnz(sel);
    if cnt(i)==3, wuxiaQ(i) = mean(ws(sel)); end     % require full quarter
end
last_full = max(uq(cnt==3));
fprintf('Wu-Xia last fully covered quarter: %s\n', datestr(last_full,'yyyy-mm'));
Tw = table(uq, wuxiaQ, 'VariableNames',{'date','WUXIA_Q'});
G  = outerjoin(G, Tw, 'Keys','date','MergeKeys',true,'Type','left');

% ------------------------------------------------ splice policy rate ------
% SHADOW_SPLICED = WUXIA_Q where available (<= last_full), else FEDFUNDS.
G.SHADOW_SPLICED = G.WUXIA_Q;
useffr = isnan(G.SHADOW_SPLICED);
G.SHADOW_SPLICED(useffr) = G.FEDFUNDS(useffr);

% ------------------------------------------------ tidy + write ------------
G = sortrows(G,'date');
G = G(:, {'date','GDPC1','GDPPOT','GDPDEF','PCEPILFE','CPIAUCSL', ...
          'FEDFUNDS','WUXIA_Q','SHADOW_SPLICED'});
G.date = datestr(G.date,'yyyy-mm-dd');
writetable(G, 'data_nk.csv');
fprintf('\nwrote data_nk.csv : %d rows x %d cols\n', height(G), width(G));

% ------------------------------------------------ splice check ------------
fprintf('\n--- splice check around 2021Q3 .. 2022Q3 ---\n');
G2 = readtable('data_nk.csv','TextType','string');
d2 = datetime(G2.date);
sel = d2>=datetime(2021,7,1) & d2<=datetime(2022,7,1);
disp(G2(sel, {'date','WUXIA_Q','FEDFUNDS','SHADOW_SPLICED'}));
