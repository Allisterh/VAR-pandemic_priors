function nk_posterior_fig(chain_nd, chain_wd, outstem, th_true)
% NK_POSTERIOR_FIG  2x4 overlaid posterior-density figure (no-dummy vs with-dummy).
%
%   nk_posterior_fig(chain_nd, chain_wd, outstem)            % US data (no truth)
%   nk_posterior_fig(chain_nd, chain_wd, outstem, th_true)   % simulated: overlay truth
%
%   Draws kernel densities of the 8 structural parameters for the two
%   specifications. If th_true (1-by-8) is given, a vertical dotted line marks
%   each parameter's true value (simulated-data version). Saves <outstem>.png/.pdf.
%
%   Cosmetics: no super-title; legend Location 'best', box off. Displays only the
%   Greek parameter names passed by the caller (no data-series mnemonics).

nm = {'\kappa','\psi_\pi','\rho_a','\rho_s','\rho_m','\sigma_a','\sigma_s','\sigma_m'};
co = [0.85 0.10 0.10; 0.00 0.45 0.85];   % no-dummy / with-dummy
ctrue = [0.15 0.15 0.15];
has_true = nargin >= 4 && ~isempty(th_true);

fig = figure('Position',[60 60 1250 620],'Color','w','Visible','off');
for k = 1:8
    subplot(2,4,k); hold on; box on;
    [f1,x1] = local_ksd(chain_nd(:,k));
    [f2,x2] = local_ksd(chain_wd(:,k));
    h1 = fill([x1 fliplr(x1)],[f1 zeros(size(f1))],co(1,:),'FaceAlpha',0.18,'EdgeColor','none');
    h2 = fill([x2 fliplr(x2)],[f2 zeros(size(f2))],co(2,:),'FaceAlpha',0.18,'EdgeColor','none');
    set([h1 h2],'HandleVisibility','off');            % keep fills out of legend
    p1 = plot(x1,f1,'--','Color',co(1,:),'LineWidth',1.6);
    p2 = plot(x2,f2,'-','Color',co(2,:),'LineWidth',1.6);
    ph = [];
    if has_true
        yl = ylim;
        ph = plot([th_true(k) th_true(k)], [0 yl(2)], ':', 'Color',ctrue,'LineWidth',1.4);
    end
    title(['$' nm{k} '$']); set(gca,'YTick',[]);
    if k == 1
        if has_true
            legend([p1 p2 ph], {'no dummy','with dummy','true'}, 'Location','best');
        else
            legend([p1 p2], {'no dummy','with dummy'}, 'Location','best');
        end
        legend boxoff;
    end
end

exportgraphics(fig, [outstem '.png'], 'Resolution',150);
exportgraphics(fig, [outstem '.pdf'], 'ContentType','vector');
close(fig);
fprintf('figure saved: %s.png\n', outstem);
end

% -------------------------------------------------------------------------
function [f, xi] = local_ksd(x)
% LOCAL_KSD  Gaussian kernel density on 100 points, matching ksdensity's
%   defaults closely enough for these posterior plots. Avoids the Statistics
%   & Machine Learning Toolbox (drop-in for ksdensity(x) with default output).
x = x(:); x = x(~isnan(x)); n = numel(x);
s  = std(x);
iqr_x = diff(prctile_local(x,[25 75]));
sig = min(s, iqr_x/1.349);                 % robust scale (as ksdensity)
if sig <= 0 || ~isfinite(sig); sig = max(s, eps); end
h  = sig * (4/(3*n))^(1/5);                % Silverman's rule of thumb
lo = min(x) - 3*h; hi = max(x) + 3*h;
xi = linspace(lo, hi, 100);
u  = (xi - x) / h;                          % n-by-100 via implicit expansion
f  = sum(exp(-0.5*u.^2), 1) / (n*h*sqrt(2*pi));
end

% -------------------------------------------------------------------------
function q = prctile_local(x, p)
% PRCTILE_LOCAL  Percentiles without the Statistics Toolbox (linear interp on
%   the (i-0.5)/n plotting positions, matching MATLAB's prctile convention).
x = sort(x(:)); n = numel(x);
if n == 1; q = repmat(x, size(p)); return; end
pos = 100*((1:n)' - 0.5)/n;
q = interp1(pos, x, p(:), 'linear');
q(p(:) <= pos(1))   = x(1);
q(p(:) >= pos(end)) = x(end);
q = reshape(q, size(p));
end
