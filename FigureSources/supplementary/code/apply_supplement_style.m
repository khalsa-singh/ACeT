function apply_supplement_style(ax, s)
%APPLY_SUPPLEMENT_STYLE Apply shared MLHealth Supplement axis formatting.
if nargin < 2 || isempty(s)
    s = supplement_style();
end
set(ax, ...
    'FontName', s.fontName, ...
    'FontSize', s.tickFontSize, ...
    'LineWidth', s.axisLineWidth, ...
    'TickDir', 'out', ...
    'Box', 'off', ...
    'Layer', 'top');
try
    ax.XGrid = 'on';
    ax.YGrid = 'on';
    ax.GridColor = s.gridColor;
    ax.GridAlpha = s.gridAlpha;
    ax.MinorGridAlpha = 0.15;
catch
end
end
