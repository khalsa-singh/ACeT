function export_supplement_figure(fig, outBase)
%EXPORT_SUPPLEMENT_FIGURE Export one figure in submission/review formats.
%   outBase is the full path without extension.

outDir = fileparts(outBase);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
set(fig, 'Color', 'w', 'InvertHardcopy', 'off');
drawnow;

% Raster review and production backups.
print(fig, [outBase '.png'], '-dpng', '-r600');
print(fig, [outBase '_RGB.tif'], '-dtiff', '-r600');

% Vector outputs.  Set painters only for vector export because image-based
% composites (e.g., S5) otherwise render poorly in the on-screen figure.
oldRenderer = get(fig, 'Renderer');
try
    set(fig, 'Renderer', 'painters');
    print(fig, [outBase '.svg'], '-dsvg', '-vector');
    print(fig, [outBase '.eps'], '-depsc2', '-painters', '-loose');
    exportgraphics(fig, [outBase '.pdf'], ...
        'ContentType', 'vector', 'BackgroundColor', 'white');
catch vectorErr
    warning('Vector export fallback for %s: %s', outBase, vectorErr.message);
    try
        print(fig, [outBase '.svg'], '-dsvg');
    catch
    end
    try
        print(fig, [outBase '.eps'], '-depsc2', '-loose');
    catch
    end
    try
        exportgraphics(fig, [outBase '.pdf'], ...
            'ContentType', 'image', 'Resolution', 600, ...
            'BackgroundColor', 'white');
    catch
    end
end
set(fig, 'Renderer', oldRenderer);

savefig(fig, [outBase '.fig']);
end
