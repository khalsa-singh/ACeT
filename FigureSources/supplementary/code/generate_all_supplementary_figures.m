function generate_all_supplementary_figures(packageRoot, outRoot)
%GENERATE_ALL_SUPPLEMENTARY_FIGURES Standardize MLHealth Figures S1-S11.
%
% Usage:
%   generate_all_supplementary_figures
%   generate_all_supplementary_figures(packageRoot)
%   generate_all_supplementary_figures(packageRoot, outRoot)
%
% The function performs plotting only. It never trains a model and never
% edits the manuscript or Supplement. Figures S1-S4 use locked robustness
% CSVs. Figures S5-S11 use already completed safe/current sources.

thisDir = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(packageRoot)
    packageRoot = locate_package_root(thisDir);
end
packageRoot = char(packageRoot);
assert(exist(fullfile(packageRoot,'MANIFEST.csv'), 'file') == 2, ...
    'Package root must contain MANIFEST.csv: %s', packageRoot);
assert(exist(fullfile(packageRoot,'Data'), 'dir') == 7, ...
    'Package root must contain Data/: %s', packageRoot);

if nargin < 2 || isempty(outRoot)
    outRoot = fullfile(packageRoot, 'ReproducedOutputs', 'SupplementaryFigures');
else
    outRoot = char(outRoot);
end
if ~exist(outRoot, 'dir'); mkdir(outRoot); end

addpath(thisDir);
s = supplement_style();

splitDir = fullfile(packageRoot, 'GenBench', 'robustness', ...
    'split_sensitivity', 'results', 'locked');
fixedDir = fullfile(packageRoot, 'GenBench', 'robustness', ...
    'fixed_holdout_stability', 'results', 'locked');
synthDir = fullfile(packageRoot, 'GenBench', 'diagnostics', ...
    'gaussian_copula', 'results');
poolDir = fullfile(packageRoot, 'GenBench', 'diagnostics', ...
    'pooling_sensitivity', 'results');
clinicalS5Dir = fullfile(packageRoot, 'FigureSources', 'supplementary', ...
    'source_data', 'clinical_s5');
hicAssaysPath = fullfile(packageRoot, 'GenBench', 'hic_benchmark', 'results', ...
    'benchmark', 'oof_predictions', 'assays_only', ...
    'hic_rt_min_OOF_assays_only_oof_predictions.csv');
hicPatchPath = fullfile(packageRoot, 'GenBench', 'hic_benchmark', 'results', ...
    'benchmark', 'oof_predictions', 'patch_only', ...
    'hic_rt_min_OOF_patch_only_oof_predictions.csv');
viscosityDataPath = fullfile(packageRoot, 'Data', 'curated', ...
    'DataS1_viscosity_seed0_full.csv');

logFile = fullfile(outRoot, 'FIGURE_GENERATION_LOG.txt');
fid = fopen(logFile, 'a');
cleanupObj = onCleanup(@() fclose_if_open(fid));
log_line(fid, '');
log_line(fid, 'Run started: %s', datestr(now, 31));
log_line(fid, 'Package root: %s', packageRoot);

% S1-S4 combine two distinct robustness questions:
%   (i) fixed manuscript holdout, independent training seeds; and
%   (ii) composite-split sensitivity with changing held-out antibodies.
visSplit = fullfile(splitDir, 'viscosity_alternate_partition_predictions.csv');
clrSplit = fullfile(splitDir, 'mouse_exposure_alternate_partition_predictions.csv');
visFixed = fullfile(fixedDir, 'viscosity_fixed_holdout_predictions.csv');
clrFixed = fullfile(fixedDir, 'clearance_fixed_holdout_predictions.csv');
requiredRobustness = {visSplit, clrSplit, visFixed, clrFixed};
if all(cellfun(@(p) exist(p, 'file') == 2, requiredRobustness))
    log_line(fid, ['Generating Figures S1-S4 from fixed-holdout seed-stability ' ...
        'and composite-split sensitivity outputs.']);
    make_s1(visFixed, visSplit, outRoot, s);
    make_s2(visFixed, visSplit, outRoot, s);
    make_s3(clrFixed, clrSplit, outRoot, s);
    make_s4(clrFixed, clrSplit, outRoot, s);
else
    log_line(fid, ['Figures S1-S4 skipped: combined robustness sources are missing. ' ...
        'See the package robustness README for locked-source locations.']);
end

% S5-S11 are no-training plots from existing sources.
log_line(fid, 'Generating Figure S5.');
make_s5(clinicalS5Dir, outRoot, s);

log_line(fid, 'Generating Figures S6-S8.');
make_s6(hicAssaysPath, outRoot, s);
make_s7(hicPatchPath, outRoot, s);
make_s8(hicAssaysPath, hicPatchPath, outRoot, s);

log_line(fid, 'Generating Figure S9.');
make_s9(viscosityDataPath, outRoot, s);

log_line(fid, 'Generating Figure S10.');
make_s10(synthDir, outRoot, s);

log_line(fid, 'Generating Figure S11.');
make_s11(poolDir, outRoot, s);

write_source_manifest(packageRoot, outRoot, visSplit, clrSplit, visFixed, ...
    clrFixed, clinicalS5Dir, hicAssaysPath, hicPatchPath, ...
    viscosityDataPath, synthDir, poolDir);
write_output_checksums(outRoot);
log_line(fid, 'Run completed: %s', datestr(now, 31));
clear cleanupObj;
fclose_if_open(fid);

fprintf('\nSupplementary figures written to:\n%s\n', outRoot);
end

%% Figure S1
function make_s1(fixedPath, splitPath, outRoot, s)
% Viscosity robustness in two complementary designs:
% a, five independent training seeds on the unchanged 52/23 manuscript split;
% b, six composite-stratified partitions with changing held-out antibodies.
F = readtable(fixedPath, 'VariableNamingRule', 'preserve');
S = readtable(splitPath, 'VariableNamingRule', 'preserve');
assert(all(F.true_value > 0 & F.predicted_value > 0), ...
    'S1 fixed-holdout viscosity values must be positive.');
assert(all(S.true_value > 0 & S.predicted_value > 0), ...
    'S1 split-sensitivity viscosity values must be positive.');

fig = figure('Color','w','Units','centimeters','Position',[2 2 17.5 9.3]);
tl = tiledlayout(fig,1,2,'TileSpacing','loose','Padding','compact');

% a, fixed-holdout training-seed stability.
ax1 = nexttile(tl,1); hold(ax1,'on');
seeds = unique(F.training_seed)';
maxErr = 0;
seedErr = cell(numel(seeds),1);
seedMetrics = zeros(numel(seeds),4); % R2, RMSE, MAE, Spearman
hThin = gobjects(1); hMean = gobjects(1);
for ii = 1:numel(seeds)
    idx = F.training_seed == seeds(ii);
    yt = F.true_value(idx); yp = F.predicted_value(idx);
    e = abs(log10(yp) - log10(yt));
    seedErr{ii} = e(:); maxErr = max(maxErr,max(e));
    [x,y] = empirical_cdf(e);
    c = mix_color(s.blue,[1 1 1],0.68);
    h = plot(ax1,x,y,'Color',c,'LineWidth',1.0,'HandleVisibility','off');
    if ii == 1; hThin = h; end
    M = regression_metrics(yt,yp);
    seedMetrics(ii,:) = [M.r2,M.rmse,M.mae,M.spearman];
end
gridX = linspace(0,max(0.35,maxErr*1.04),240)';
cdfMatrix = zeros(numel(gridX),numel(seeds));
for ii = 1:numel(seeds)
    cdfMatrix(:,ii) = arrayfun(@(q) mean(seedErr{ii} <= q)*100, gridX);
end
hMean = plot(ax1,gridX,mean(cdfMatrix,2),'Color',s.blue, ...
    'LineWidth',s.mainLineWidth,'DisplayName','Mean CDF');
x2 = log10(2); x3 = log10(3);
xline(ax1,x2,':','Color',s.gray,'LineWidth',1.0,'HandleVisibility','off');
xline(ax1,x3,':','Color',s.gray,'LineWidth',1.0,'HandleVisibility','off');
metricsText = sprintf(['R^2 = %.3f \\pm %.3f\nRMSE = %.2f \\pm %.2f cP\n' ...
    'Spearman \\rho = %.3f \\pm %.3f'], ...
    mean(seedMetrics(:,1)),std(seedMetrics(:,1),0), ...
    mean(seedMetrics(:,2)),std(seedMetrics(:,2),0), ...
    mean(seedMetrics(:,4)),std(seedMetrics(:,4),0));
annotation_text_at_right(ax1,metricsText,[0.96 0.60],s);
title(ax1,'Fixed holdout: five training seeds','FontName',s.fontName, ...
    'FontSize',s.labelFontSize,'FontWeight','normal');
xlabel(ax1,'Absolute log_{10} error','FontName',s.fontName, ...
    'FontSize',s.labelFontSize,'FontWeight','normal');
ylabel(ax1,'Cumulative percentage','FontName',s.fontName, ...
    'FontSize',s.labelFontSize,'FontWeight','normal');
xlim(ax1,[0 max(gridX)]); ylim(ax1,[0 100]);
apply_supplement_style(ax1,s); panel_label(ax1,'a',s);
legend(ax1,[hThin,hMean],{'Individual seeds','Mean CDF'}, ...
    'Location','southeast','Box','off','FontName',s.fontName, ...
    'FontSize',s.legendFontSize); hold(ax1,'off');

% b, composite-split sensitivity; no aggregate R2 across changing test sets.
ax2 = nexttile(tl,2); hold(ax2,'on');
splitSeeds = unique(S.split_seed)';
allErr = abs(log10(S.predicted_value)-log10(S.true_value));
hOther = gobjects(1); hSeed0 = gobjects(1); hPooled = gobjects(1);
for ii = 1:numel(splitSeeds)
    idx = S.split_seed == splitSeeds(ii);
    [x,y] = empirical_cdf(allErr(idx));
    if splitSeeds(ii) == 0
        hSeed0 = plot(ax2,x,y,'Color',s.blue,'LineWidth',1.5, ...
            'DisplayName','Manuscript partition');
    else
        h = plot(ax2,x,y,'Color',s.lightGray,'LineWidth',1.0, ...
            'HandleVisibility','off');
        if ~isgraphics(hOther); hOther = h; end
    end
end
[xP,yP] = empirical_cdf(allErr);
hPooled = plot(ax2,xP,yP,'Color',s.black,'LineWidth',s.mainLineWidth, ...
    'DisplayName','All prediction instances');
xline(ax2,x2,':','Color',s.gray,'LineWidth',1.0,'HandleVisibility','off');
xline(ax2,x3,':','Color',s.gray,'LineWidth',1.0,'HandleVisibility','off');
splitText = sprintf(['n = %d prediction instances\nWithin 2-fold: %.1f%%\n' ...
    'Within 3-fold: %.1f%%'],numel(allErr),mean(allErr<=x2)*100, ...
    mean(allErr<=x3)*100);
annotation_text_at_right(ax2,splitText,[0.96 0.60],s);
title(ax2,'Alternate composite partitions','FontName',s.fontName, ...
    'FontSize',s.labelFontSize,'FontWeight','normal');
xlabel(ax2,'Absolute log_{10} error','FontName',s.fontName, ...
    'FontSize',s.labelFontSize,'FontWeight','normal');
xlim(ax2,[0 max(0.5,max(allErr)*1.04)]); ylim(ax2,[0 100]);
apply_supplement_style(ax2,s); panel_label(ax2,'b',s);
if ~isgraphics(hOther)
    hOther = plot(ax2,nan,nan,'Color',s.lightGray,'LineWidth',1.0);
end
legend(ax2,[hOther,hSeed0,hPooled], ...
    {'Other partitions','Manuscript partition','All prediction instances'}, ...
    'Location','southeast','Box','off','FontName',s.fontName, ...
    'FontSize',s.legendFontSize); hold(ax2,'off');

export_supplement_figure(fig,fullfile(outRoot,'Figure_S1_Viscosity_Robustness_CDF'));
close(fig);
end

%% Figure S2
function make_s2(fixedPath, splitPath, outRoot, s)
% Viscosity parity under a fixed holdout (a) and changing composite splits (b).
F = readtable(fixedPath,'VariableNamingRule','preserve');
S = readtable(splitPath,'VariableNamingRule','preserve');
fig = figure('Color','w','Units','centimeters','Position',[2 2 17.5 10.2]);
tl = tiledlayout(fig,1,2,'TileSpacing','loose','Padding','compact');

% a, same 23 antibodies across five training seeds.
ax1 = nexttile(tl,1); hold(ax1,'on');
[ids,~,group] = unique(string(F.mAb_id),'stable');
trueVals = accumarray(group,F.true_value,[],@mean);
meanPred = accumarray(group,F.predicted_value,[],@mean);
sdPred = accumarray(group,F.predicted_value,[],@(x) std(x,0));
for ii = 1:numel(ids)
    idx = group == ii;
    scatter(ax1,repmat(trueVals(ii),sum(idx),1),F.predicted_value(idx), ...
        16,'o','MarkerFaceColor',s.lightGray,'MarkerEdgeColor','none', ...
        'MarkerFaceAlpha',0.48,'HandleVisibility','off');
end
hMean = errorbar(ax1,trueVals,meanPred,sdPred,'o','LineStyle','none', ...
    'Color',s.blue,'MarkerFaceColor','w','MarkerEdgeColor',s.blue, ...
    'LineWidth',1.2,'CapSize',5,'MarkerSize',5.5, ...
    'DisplayName','Across-seed mean \\pm s.d.');
idx0 = F.training_seed == 0;
hSeed0 = scatter(ax1,F.true_value(idx0),F.predicted_value(idx0), ...
    s.markerSize,'d','MarkerFaceColor',s.blue,'MarkerEdgeColor',s.black, ...
    'LineWidth',0.6,'DisplayName','Seed 0');
vals = [F.true_value;F.predicted_value]; lo=max(1,min(vals)*0.88); hi=max(vals)*1.12;
shade_viscosity_quadrants(ax1,lo,hi,20);
hUnity = plot(ax1,[lo hi],[lo hi],'--','Color',s.black, ...
    'LineWidth',s.secondaryLineWidth,'DisplayName','Unity');
xline(ax1,15,':','Color',s.gray,'LineWidth',0.9,'HandleVisibility','off');
yline(ax1,15,':','Color',s.gray,'LineWidth',0.9,'HandleVisibility','off');
xline(ax1,20,'-','Color',s.gray,'LineWidth',0.9,'HandleVisibility','off');
yline(ax1,20,'-','Color',s.gray,'LineWidth',0.9,'HandleVisibility','off');
set(ax1,'XScale','log','YScale','log'); xlim(ax1,[lo hi]); ylim(ax1,[lo hi]); axis(ax1,'square');
xlabel(ax1,'Observed viscosity (cP)','FontName',s.fontName,'FontSize',s.labelFontSize,'FontWeight','normal');
ylabel(ax1,'Predicted viscosity (cP)','FontName',s.fontName,'FontSize',s.labelFontSize,'FontWeight','normal');
apply_supplement_style(ax1,s); panel_label_inside(ax1,'a',s);
legend(ax1,[hMean,hSeed0,hUnity],{'Mean \pm s.d.','Seed 0','Unity'}, ...
    'Location','northoutside','Orientation','horizontal','NumColumns',3, ...
    'Box','off','Interpreter','tex','FontName',s.fontName, ...
    'FontSize',s.legendFontSize-1);
hold(ax1,'off');

% b, different held-out antibodies across six composite partitions.
ax2 = nexttile(tl,2); hold(ax2,'on');
vals = [S.true_value;S.predicted_value]; lo=max(1,min(vals)*0.88); hi=max(vals)*1.12;
shade_viscosity_quadrants(ax2,lo,hi,20);
hSeed0b = gobjects(1);
for seed = unique(S.split_seed)'
    idx = S.split_seed == seed;
    if seed == 0
        hSeed0b = scatter(ax2,S.true_value(idx),S.predicted_value(idx), ...
            s.largeMarkerSize,'d','MarkerFaceColor',s.blue, ...
            'MarkerEdgeColor',s.black,'LineWidth',0.7, ...
            'DisplayName','Manuscript partition');
    else
        scatter(ax2,S.true_value(idx),S.predicted_value(idx),s.markerSize,'o', ...
            'MarkerFaceColor',s.lightGray,'MarkerEdgeColor',[0.72 0.72 0.72], ...
            'MarkerFaceAlpha',0.55,'MarkerEdgeAlpha',0.35,'HandleVisibility','off');
    end
end
hOther = scatter(ax2,nan,nan,s.markerSize,'o','MarkerFaceColor',s.lightGray, ...
    'MarkerEdgeColor',[0.72 0.72 0.72],'DisplayName','Other partitions');
hUnityb = plot(ax2,[lo hi],[lo hi],'--','Color',s.black, ...
    'LineWidth',s.secondaryLineWidth,'DisplayName','Unity');
xline(ax2,15,':','Color',s.gray,'LineWidth',0.9,'HandleVisibility','off');
yline(ax2,15,':','Color',s.gray,'LineWidth',0.9,'HandleVisibility','off');
xline(ax2,20,'-','Color',s.gray,'LineWidth',0.9,'HandleVisibility','off');
yline(ax2,20,'-','Color',s.gray,'LineWidth',0.9,'HandleVisibility','off');
set(ax2,'XScale','log','YScale','log'); xlim(ax2,[lo hi]); ylim(ax2,[lo hi]); axis(ax2,'square');
xlabel(ax2,'Observed viscosity (cP)','FontName',s.fontName,'FontSize',s.labelFontSize,'FontWeight','normal');
ylabel(ax2,'Predicted viscosity (cP)','FontName',s.fontName,'FontSize',s.labelFontSize,'FontWeight','normal');
apply_supplement_style(ax2,s); panel_label_inside(ax2,'b',s);
legend(ax2,[hOther,hSeed0b,hUnityb],{'Other splits','Manuscript split','Unity'}, ...
    'Location','northoutside','Orientation','horizontal','NumColumns',3, ...
    'Box','off','FontName',s.fontName,'FontSize',s.legendFontSize-1);
hold(ax2,'off');

export_supplement_figure(fig,fullfile(outRoot,'Figure_S2_Viscosity_Robustness_Parity'));
close(fig);
end

%% Figure S3
function make_s3(fixedPath, splitPath, outRoot, s)
% Mouse-exposure absolute-percentage-error stability under fixed holdout (a)
% and changing composite partitions (b).
F = readtable(fixedPath,'VariableNamingRule','preserve');
S = readtable(splitPath,'VariableNamingRule','preserve');
fig = figure('Color','w','Units','centimeters','Position',[2 2 17.5 9.3]);
tl = tiledlayout(fig,1,2,'TileSpacing','loose','Padding','compact');

ax1 = nexttile(tl,1); hold(ax1,'on');
seeds = unique(F.training_seed)';
seedErr = cell(numel(seeds),1); maxErr=0; seedMetrics=zeros(numel(seeds),4);
hThin=gobjects(1); hMean=gobjects(1);
for ii=1:numel(seeds)
    idx=F.training_seed==seeds(ii); yt=F.true_value(idx); yp=F.predicted_value(idx);
    e=abs((yp-yt)./yt).*100; seedErr{ii}=e(:); maxErr=max(maxErr,max(e));
    [x,y]=empirical_cdf(e); c=mix_color(s.orange,[1 1 1],0.68);
    h=plot(ax1,x,y,'Color',c,'LineWidth',1.0,'HandleVisibility','off');
    if ii==1;hThin=h;end
    M=regression_metrics(yt,yp);
    seedMetrics(ii,:)=[M.r2,M.rmse/mean(yt),M.mae/mean(yt),M.spearman];
end
gridX=linspace(0,max(55,maxErr*1.04),240)'; cdfMatrix=zeros(numel(gridX),numel(seeds));
for ii=1:numel(seeds);cdfMatrix(:,ii)=arrayfun(@(q)mean(seedErr{ii}<=q)*100,gridX);end
hMean=plot(ax1,gridX,mean(cdfMatrix,2),'Color',s.orange,'LineWidth',s.mainLineWidth,'DisplayName','Mean CDF');
xline(ax1,15,':','Color',s.gray,'LineWidth',1.0,'HandleVisibility','off');
xline(ax1,30,':','Color',s.gray,'LineWidth',1.0,'HandleVisibility','off');
metricsText=sprintf(['R^2 = %.3f \\pm %.3f\nnRMSE = %.3f \\pm %.3f\n' ...
    'Spearman \\rho = %.3f \\pm %.3f'], ...
    mean(seedMetrics(:,1)),std(seedMetrics(:,1),0), ...
    mean(seedMetrics(:,2)),std(seedMetrics(:,2),0), ...
    mean(seedMetrics(:,4)),std(seedMetrics(:,4),0));
annotation_text_at_right(ax1,metricsText,[0.96 0.60],s);
title(ax1,'Fixed holdout: five training seeds','FontName',s.fontName, ...
    'FontSize',s.labelFontSize,'FontWeight','normal');
xlabel(ax1,'Absolute percentage error','FontName',s.fontName,'FontSize',s.labelFontSize,'FontWeight','normal');
ylabel(ax1,'Cumulative percentage','FontName',s.fontName,'FontSize',s.labelFontSize,'FontWeight','normal');
xlim(ax1,[0 max(gridX)]);ylim(ax1,[0 100]);apply_supplement_style(ax1,s);panel_label(ax1,'a',s);
legend(ax1,[hThin,hMean],{'Individual seeds','Mean CDF'},'Location','southeast', ...
    'Box','off','FontName',s.fontName,'FontSize',s.legendFontSize);hold(ax1,'off');

ax2=nexttile(tl,2);hold(ax2,'on');
allErr=abs((S.predicted_value-S.true_value)./S.true_value).*100;
hOther=gobjects(1);hSeed0=gobjects(1);hPooled=gobjects(1);
for seed=unique(S.split_seed)'
    idx=S.split_seed==seed;[x,y]=empirical_cdf(allErr(idx));
    if seed==0
        hSeed0=plot(ax2,x,y,'Color',s.orange,'LineWidth',1.5,'DisplayName','Manuscript partition');
    else
        h=plot(ax2,x,y,'Color',s.lightGray,'LineWidth',1.0,'HandleVisibility','off');
        if ~isgraphics(hOther);hOther=h;end
    end
end
[xP,yP]=empirical_cdf(allErr);hPooled=plot(ax2,xP,yP,'Color',s.black,'LineWidth',s.mainLineWidth,'DisplayName','All prediction instances');
xline(ax2,15,':','Color',s.gray,'LineWidth',1.0,'HandleVisibility','off');xline(ax2,30,':','Color',s.gray,'LineWidth',1.0,'HandleVisibility','off');
splitText=sprintf(['n = %d prediction instances\nWithin 15%%: %.1f%%\nWithin 30%%: %.1f%%'], ...
    numel(allErr),mean(allErr<=15)*100,mean(allErr<=30)*100);
annotation_text_at_right(ax2,splitText,[0.96 0.60],s);
title(ax2,'Alternate composite partitions','FontName',s.fontName, ...
    'FontSize',s.labelFontSize,'FontWeight','normal');
xlabel(ax2,'Absolute percentage error','FontName',s.fontName,'FontSize',s.labelFontSize,'FontWeight','normal');
xlim(ax2,[0 max(60,max(allErr)*1.03)]);ylim(ax2,[0 100]);apply_supplement_style(ax2,s);panel_label(ax2,'b',s);
if ~isgraphics(hOther);hOther=plot(ax2,nan,nan,'Color',s.lightGray,'LineWidth',1.0);end
legend(ax2,[hOther,hSeed0,hPooled],{'Other partitions','Manuscript partition','All prediction instances'}, ...
    'Location','southeast','Box','off','FontName',s.fontName,'FontSize',s.legendFontSize);hold(ax2,'off');

export_supplement_figure(fig,fullfile(outRoot,'Figure_S3_Clearance_Robustness_CDF_Jackknife'));
close(fig);
end

%% Figure S4
function make_s4(fixedPath, splitPath, outRoot, s)
% Mouse-exposure parity under fixed holdout (a) and changing composite splits (b).
% Values are displayed directly in millions to avoid duplicated scientific-
% notation multipliers and unit-label overlap.
F=readtable(fixedPath,'VariableNamingRule','preserve');
S=readtable(splitPath,'VariableNamingRule','preserve');
fig=figure('Color','w','Units','centimeters','Position',[2 2 17.5 10.2]);
tl=tiledlayout(fig,1,2,'TileSpacing','loose','Padding','compact');
storedToOriginal=1e4;
displayScale=storedToOriginal/1e6;
thr=3.9;

ax1=nexttile(tl,1);hold(ax1,'on');
[ids,~,group]=unique(string(F.mAb_id),'stable');
trueVals=accumarray(group,F.true_value,[],@mean).*displayScale;
meanPred=accumarray(group,F.predicted_value,[],@mean).*displayScale;
sdPred=accumarray(group,F.predicted_value,[],@(x)std(x,0)).*displayScale;
for ii=1:numel(ids)
    idx=group==ii;
    scatter(ax1,repmat(trueVals(ii),sum(idx),1),F.predicted_value(idx).*displayScale, ...
        16,'o','MarkerFaceColor',s.lightGray,'MarkerEdgeColor','none', ...
        'MarkerFaceAlpha',0.48,'HandleVisibility','off');
end
hMean=errorbar(ax1,trueVals,meanPred,sdPred,'o','LineStyle','none','Color',s.orange, ...
    'MarkerFaceColor','w','MarkerEdgeColor',s.orange,'LineWidth',1.2,'CapSize',5, ...
    'MarkerSize',5.5,'DisplayName','Across-seed mean \\pm s.d.');
idx0=F.training_seed==0;
hSeed0=scatter(ax1,F.true_value(idx0).*displayScale,F.predicted_value(idx0).*displayScale, ...
    s.markerSize,'o','MarkerFaceColor',s.orange,'MarkerEdgeColor',s.black, ...
    'LineWidth',0.6,'DisplayName','Seed 0');
vals=[F.true_value;F.predicted_value].*displayScale;lo=max(0.1,min(vals)*0.88);hi=max(vals)*1.12;
shade_clearance_quadrants(ax1,lo,hi,thr);
hUnity=plot(ax1,[lo hi],[lo hi],'--','Color',s.black,'LineWidth',s.secondaryLineWidth,'DisplayName','Unity');
xline(ax1,thr,':','Color',s.gray,'LineWidth',0.9,'HandleVisibility','off');yline(ax1,thr,':','Color',s.gray,'LineWidth',0.9,'HandleVisibility','off');
set(ax1,'XScale','log','YScale','log');xlim(ax1,[lo hi]);ylim(ax1,[lo hi]);axis(ax1,'square');
ax1.XTick=[2 4 8 12];ax1.YTick=[2 4 8 12];
xlabel(ax1,'Observed AUC_t (10^6 ng\cdot h mL^{-1})','Interpreter','tex','FontName',s.fontName,'FontSize',s.labelFontSize,'FontWeight','normal');
ylabel(ax1,'Predicted AUC_t (10^6 ng\cdot h mL^{-1})','Interpreter','tex','FontName',s.fontName,'FontSize',s.labelFontSize,'FontWeight','normal');
apply_supplement_style(ax1,s);panel_label_inside(ax1,'a',s);
legend(ax1,[hMean,hSeed0,hUnity],{'Mean \pm s.d.','Seed 0','Unity'}, ...
    'Location','northoutside','Orientation','horizontal','NumColumns',3, ...
    'Box','off','Interpreter','tex','FontName',s.fontName, ...
    'FontSize',s.legendFontSize-1);hold(ax1,'off');

ax2=nexttile(tl,2);hold(ax2,'on');
xAll=S.true_value.*displayScale;yAll=S.predicted_value.*displayScale;vals=[xAll;yAll];lo=max(0.1,min(vals)*0.88);hi=max(vals)*1.12;
shade_clearance_quadrants(ax2,lo,hi,thr);hSeed0b=gobjects(1);
for seed=unique(S.split_seed)'
    idx=S.split_seed==seed;
    if seed==0
        hSeed0b=scatter(ax2,xAll(idx),yAll(idx),s.largeMarkerSize,'o','MarkerFaceColor',s.orange, ...
            'MarkerEdgeColor',s.black,'LineWidth',0.7,'DisplayName','Manuscript partition');
    else
        scatter(ax2,xAll(idx),yAll(idx),s.markerSize,'o','MarkerFaceColor',s.lightGray, ...
            'MarkerEdgeColor',[0.72 0.72 0.72],'MarkerFaceAlpha',0.55, ...
            'MarkerEdgeAlpha',0.35,'HandleVisibility','off');
    end
end
hOther=scatter(ax2,nan,nan,s.markerSize,'o','MarkerFaceColor',s.lightGray, ...
    'MarkerEdgeColor',[0.72 0.72 0.72],'DisplayName','Other partitions');
hUnityb=plot(ax2,[lo hi],[lo hi],'--','Color',s.black,'LineWidth',s.secondaryLineWidth,'DisplayName','Unity');
xline(ax2,thr,':','Color',s.gray,'LineWidth',0.9,'HandleVisibility','off');yline(ax2,thr,':','Color',s.gray,'LineWidth',0.9,'HandleVisibility','off');
set(ax2,'XScale','log','YScale','log');xlim(ax2,[lo hi]);ylim(ax2,[lo hi]);axis(ax2,'square');
ax2.XTick=[2 4 8 12];ax2.YTick=[2 4 8 12];
xlabel(ax2,'Observed AUC_t (10^6 ng\cdot h mL^{-1})','Interpreter','tex','FontName',s.fontName,'FontSize',s.labelFontSize,'FontWeight','normal');
ylabel(ax2,'Predicted AUC_t (10^6 ng\cdot h mL^{-1})','Interpreter','tex','FontName',s.fontName,'FontSize',s.labelFontSize,'FontWeight','normal');
apply_supplement_style(ax2,s);panel_label_inside(ax2,'b',s);
legend(ax2,[hOther,hSeed0b,hUnityb],{'Other splits','Manuscript split','Unity'}, ...
    'Location','northoutside','Orientation','horizontal','NumColumns',3, ...
    'Box','off','FontName',s.fontName,'FontSize',s.legendFontSize-1);hold(ax2,'off');

export_supplement_figure(fig,fullfile(outRoot,'Figure_S4_Clearance_Robustness_Parity'));
close(fig);
end


%% Shared robustness-plot helpers restored after final formatting patch
function [x,y]=empirical_cdf(values)
values=values(:);
values=values(isfinite(values));
assert(~isempty(values),'empirical_cdf received no finite values.');
x=sort(values);
y=(1:numel(x))'./numel(x).*100;
end

function shade_viscosity_quadrants(ax,lo,hi,thr)
patch(ax,[lo thr thr lo],[lo lo thr thr],[0.88 0.96 0.88], ...
    'EdgeColor','none','FaceAlpha',0.35,'HandleVisibility','off');
patch(ax,[thr hi hi thr],[thr thr hi hi],[0.88 0.96 0.88], ...
    'EdgeColor','none','FaceAlpha',0.35,'HandleVisibility','off');
patch(ax,[lo thr thr lo],[thr thr hi hi],[0.98 0.91 0.91], ...
    'EdgeColor','none','FaceAlpha',0.35,'HandleVisibility','off');
patch(ax,[thr hi hi thr],[lo lo thr thr],[0.98 0.91 0.91], ...
    'EdgeColor','none','FaceAlpha',0.35,'HandleVisibility','off');
end

function shade_clearance_quadrants(ax,lo,hi,thr)
patch(ax,[lo thr thr lo],[lo lo thr thr],[0.88 0.96 0.88], ...
    'EdgeColor','none','FaceAlpha',0.35,'HandleVisibility','off');
patch(ax,[thr hi hi thr],[thr thr hi hi],[0.88 0.96 0.88], ...
    'EdgeColor','none','FaceAlpha',0.35,'HandleVisibility','off');
patch(ax,[lo thr thr lo],[thr thr hi hi],[0.98 0.91 0.91], ...
    'EdgeColor','none','FaceAlpha',0.35,'HandleVisibility','off');
patch(ax,[thr hi hi thr],[lo lo thr thr],[0.98 0.91 0.91], ...
    'EdgeColor','none','FaceAlpha',0.35,'HandleVisibility','off');
end

%% Figure S5
function make_s5(sourceDir, outRoot, s)
files = { ...
    fullfile(sourceDir,'clinical_external_performance.png'), ...
    fullfile(sourceDir,'clinical_external_logloss_cdf.png'), ...
    fullfile(sourceDir,'clinical_external_jackknife.png')};
assert(all(cellfun(@(p) exist(p,'file')==2,files)), ...
    'S5 source images were not found in %s',sourceDir);
fig = figure('Color','w','Units','centimeters','Position',[1 1 27.0 8.8]);
tl = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
ax1 = nexttile(tl,1); show_image_panel(ax1,files{1},'a',s);
ax2 = nexttile(tl,2); show_image_panel(ax2,files{2},'b',s);
ax3 = nexttile(tl,3); show_image_panel(ax3,files{3},'c',s);
export_supplement_figure(fig, fullfile(outRoot,'Figure_S5_Clinical_External_Robustness'));
close(fig);
end

%% Figure S6
function make_s6(csvPath, outRoot, s)
T = readtable(csvPath, 'VariableNamingRule','preserve');
trueY = T.True; predY = T.OOF_ACeT;
M = regression_metrics(trueY,predY);
[ba,mcc] = triage_metrics(trueY,predY,30);
fig = figure('Color','w','Units','centimeters','Position',[2 2 17.5 9.5]);
ax = axes('Parent',fig,'Position',[0.08 0.14 0.62 0.78]); hold(ax,'on');
scatter(ax,trueY,predY,s.markerSize,'o','MarkerFaceColor',s.blue, ...
    'MarkerEdgeColor','none','MarkerFaceAlpha',0.82);
limits = parity_limits(trueY,predY,2);
plot(ax,limits,limits,'--','Color',s.black,'LineWidth',s.secondaryLineWidth);
xlim(ax,limits); ylim(ax,limits); axis(ax,'square');
xlabel(ax,'Observed HIC retention time (min)','FontName',s.fontName, ...
    'FontSize',s.labelFontSize,'FontWeight','normal');
ylabel(ax,'Predicted HIC retention time (min)','FontName',s.fontName, ...
    'FontSize',s.labelFontSize,'FontWeight','normal');
apply_supplement_style(ax,s); hold(ax,'off');
summary=sprintf(['n = %d\nR^2 = %.3f\nPearson r^2 = %.3f\n' ...
    'Spearman \\rho = %.3f\nRMSE = %.2f min\nMAE = %.2f min\n' ...
    'BalAcc@30 = %.3f\nMCC@30 = %.3f'], ...
    numel(trueY),M.r2,M.pearson2,M.spearman,M.rmse,M.mae,ba,mcc);
figure_textbox(fig,[0.73 0.22 0.25 0.56],summary,s);
export_supplement_figure(fig, fullfile(outRoot,'Figure_S6_HIC_AssaysOnly_OOF_Parity'));
close(fig);
end

%% Figure S7
function make_s7(csvPath, outRoot, s)
T = readtable(csvPath, 'VariableNamingRule','preserve');
trueY = T.True;
series = {T.OOF_ACeT,T.OOF_BaillyRefit};
labels = {'ACeT (patch only)','Bailly refit (patch linear)'};
colors = {s.blue,s.orange};
allPred = [series{1};series{2}]; limits = parity_limits(trueY,allPred,2);
fig = figure('Color','w','Units','centimeters','Position',[2 2 17.5 12.6]);
positions={[0.07 0.46 0.40 0.49],[0.56 0.46 0.40 0.49]};
summaryPositions={[0.07 0.035 0.40 0.27],[0.56 0.035 0.40 0.27]};
for k=1:2
    ax=axes('Parent',fig,'Position',positions{k}); hold(ax,'on'); predY=series{k};
    scatter(ax,trueY,predY,s.markerSize,'o','MarkerFaceColor',colors{k}, ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',0.82);
    plot(ax,limits,limits,'--','Color',s.black,'LineWidth',s.secondaryLineWidth);
    M=regression_metrics(trueY,predY); [ba,mcc]=triage_metrics(trueY,predY,30);
    xlim(ax,limits);ylim(ax,limits);axis(ax,'square');
    title(ax,labels{k},'FontName',s.fontName,'FontSize',s.labelFontSize, ...
        'FontWeight','normal');
    xlabel(ax,'Observed HIC retention time (min)','FontName',s.fontName, ...
        'FontSize',s.labelFontSize,'FontWeight','normal');
    ylabel(ax,'Predicted HIC retention time (min)','FontName',s.fontName, ...
        'FontSize',s.labelFontSize,'FontWeight','normal');
    apply_supplement_style(ax,s); panel_label(ax,char('a'+k-1),s); hold(ax,'off');
    summary=sprintf(['n = %d\nR^2 = %.3f\nPearson r^2 = %.3f\n' ...
        'Spearman \\rho = %.3f\nRMSE = %.2f min; MAE = %.2f min\n' ...
        'BalAcc@30 = %.3f; MCC@30 = %.3f'], ...
        numel(trueY),M.r2,M.pearson2,M.spearman,M.rmse,M.mae,ba,mcc);
    figure_textbox(fig,summaryPositions{k},summary,s);
end
export_supplement_figure(fig, fullfile(outRoot,'Figure_S7_HIC_PatchOnly_Comparison'));
close(fig);
end

%% Figure S8
function make_s8(assaysPath, patchPath, outRoot, s)
A = readtable(assaysPath, 'VariableNamingRule','preserve');
P = readtable(patchPath, 'VariableNamingRule','preserve');
items = { ...
    confusion_at_threshold(A.True,A.OOF_ACeT,30), ...
    confusion_at_threshold(A.True,A.OOF_BaillyRefit,30), ...
    confusion_at_threshold(P.True,P.OOF_ACeT,30), ...
    confusion_at_threshold(P.True,P.OOF_BaillyRefit,30)};
titles = {'Assays only: ACeT','Assays only: Bailly refit', ...
    'Patch only: ACeT','Patch only: Bailly refit'};
maxC=max(cellfun(@(x) max(x(:)),items));
fig=figure('Color','w','Units','centimeters','Position',[2 2 17.5 14]);
tl=tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');
for k=1:4
    ax=nexttile(tl,k); cm=items{k}; imagesc(ax,cm); axis(ax,'equal','tight');
    colormap(ax,parula); caxis(ax,[0 maxC]); ax.YDir='reverse';
    ax.XTick=1:2; ax.XTickLabel={'Good','Poor'};
    ax.YTick=1:2; ax.YTickLabel={'Good','Poor'};
    title(ax,titles{k},'FontName',s.fontName,'FontSize',s.labelFontSize, ...
        'FontWeight','normal');
    xlabel(ax,'Predicted','FontName',s.fontName,'FontSize',s.labelFontSize, ...
        'FontWeight','normal');
    ylabel(ax,'Observed','FontName',s.fontName,'FontSize',s.labelFontSize, ...
        'FontWeight','normal');
    rowPct=cm./sum(cm,2).*100;
    for i=1:2
        for j=1:2
            text(ax,j,i,sprintf('%d\n(%.0f%%)',cm(i,j),rowPct(i,j)), ...
                'HorizontalAlignment','center','VerticalAlignment','middle', ...
                'FontName',s.fontName,'FontSize',11,'FontWeight','bold', ...
                'Color',contrast_text_color(cm(i,j),maxC));
        end
    end
    apply_supplement_style(ax,s); ax.XGrid='off'; ax.YGrid='off';
    panel_label(ax,char('a'+k-1),s);
end
cb=colorbar(nexttile(tl,4),'eastoutside'); cb.Label.String='Count';
cb.FontName=s.fontName; cb.FontSize=s.tickFontSize;
export_supplement_figure(fig, fullfile(outRoot,'Figure_S8_HIC_Triage_ConfusionMatrices'));
close(fig);
end

%% Figure S9
function make_s9(csvPath, outRoot, s)
T=readtable(csvPath,'VariableNamingRule','preserve');
plates=T.('SE-UHPLC Main Peak Plates (EP)');
fwhm=T.('SE-UHPLC Main Peak FWHM (min)');
visc=T.Viscosity;
fig=figure('Color','w','Units','centimeters','Position',[2 2 17.5 8.5]);
tl=tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
ax1=nexttile(tl,1); scatter(ax1,plates,visc,s.markerSize,'o', ...
    'MarkerFaceColor',s.blue,'MarkerEdgeColor','none','MarkerFaceAlpha',0.78);
xlabel(ax1,'SE-UHPLC plate count (EP)','FontName',s.fontName, ...
    'FontSize',s.labelFontSize,'FontWeight','normal');
ylabel(ax1,'Viscosity (cP)','FontName',s.fontName, ...
    'FontSize',s.labelFontSize,'FontWeight','normal');
apply_supplement_style(ax1,s);panel_label_inside(ax1,'a',s);
ax2=nexttile(tl,2);scatter(ax2,fwhm,visc,s.markerSize,'o', ...
    'MarkerFaceColor',s.blue,'MarkerEdgeColor','none','MarkerFaceAlpha',0.78);
xlabel(ax2,'SE-UHPLC FWHM (min)','FontName',s.fontName, ...
    'FontSize',s.labelFontSize,'FontWeight','normal');
ylabel(ax2,'Viscosity (cP)','FontName',s.fontName, ...
    'FontSize',s.labelFontSize,'FontWeight','normal');
apply_supplement_style(ax2,s);panel_label_inside(ax2,'b',s);
export_supplement_figure(fig, fullfile(outRoot,'Figure_S9_SEC_PeakShape_Viscosity'));
close(fig);
end

%% Figure S10
function make_s10(sourceDir, outRoot, s)
endpoints={'viscosity','clearance'};
columnTitles={'PCA overlap','Marginal differences',{'Absolute Spearman','correlation difference'}, ...
    'Nearest-neighbor distances'};
fig=figure('Color','w','Units','centimeters','Position',[1 1 27.0 13.5]);
tl=tiledlayout(fig,2,4,'TileSpacing','compact','Padding','compact');
letters='abcdefgh';
for row=1:2
    ep=endpoints{row};
    samples=readtable(fullfile(sourceDir,[ep '_real_vs_synthetic_samples.csv']), ...
        'VariableNamingRule','preserve');
    marginal=readtable(fullfile(sourceDir,[ep '_marginal_statistics.csv']), ...
        'VariableNamingRule','preserve');
    [X,source,names]=diagnostic_matrix(samples);
    displayNames=short_feature_names(ep,names);
    realMask=source=="real_training"; synthMask=source=="synthetic";
    realX=X(realMask,:); synthX=X(synthMask,:);
    [scores,explained]=pca_from_combined_scale(realX,synthX);

    % PCA overlap.
    ax=nexttile(tl,(row-1)*4+1);hold(ax,'on');
    scatter(ax,scores(realMask,1),scores(realMask,2),26,'o', ...
        'MarkerFaceColor',s.blue,'MarkerEdgeColor','none','MarkerFaceAlpha',0.72);
    scatter(ax,scores(synthMask,1),scores(synthMask,2),29,'d', ...
        'MarkerFaceColor',s.orange,'MarkerEdgeColor','none','MarkerFaceAlpha',0.68);
    xlabel(ax,sprintf('PC1 (%.1f%%)',explained(1)),'FontName',s.fontName, ...
        'FontSize',9.5,'FontWeight','normal');
    ylabel(ax,sprintf('PC2 (%.1f%%)',explained(2)),'FontName',s.fontName, ...
        'FontSize',9.5,'FontWeight','normal');
    apply_supplement_style(ax,s); ax.FontSize=8.5;
    panel_label_inside(ax,letters((row-1)*4+1),s);
    if row==1
        title(ax,{'PCA overlap','Viscosity'},'FontName',s.fontName, ...
            'FontSize',10.5,'FontWeight','normal');
        legend(ax,{'Real training','Synthetic'},'Location','southeast','Box','off', ...
            'FontName',s.fontName,'FontSize',7.5);
    else
        title(ax,'Mouse clearance','FontName',s.fontName,'FontSize',10.5, ...
            'FontWeight','normal');
    end
    hold(ax,'off');

    % Marginal KS statistics.
    ax=nexttile(tl,(row-1)*4+2);
    ks=marginal.ks_statistic;
    barh(ax,1:numel(ks),ks,0.65,'FaceColor',s.orange,'EdgeColor','none');
    ax.YTick=1:numel(ks); ax.YTickLabel=displayNames; ax.YDir='reverse';
    xlabel(ax,'KS statistic','FontName',s.fontName,'FontSize',9.5, ...
        'FontWeight','normal');
    xlim(ax,[0 max(0.4,max(ks)*1.15)]);apply_supplement_style(ax,s);
    ax.FontSize=8.5;
    panel_label_inside(ax,letters((row-1)*4+2),s);
    if row==1
        title(ax,columnTitles{2},'FontName',s.fontName,'FontSize',10.5, ...
            'FontWeight','normal');
    end

    % Absolute correlation difference.
    ax=nexttile(tl,(row-1)*4+3);
    corrDiff=read_numeric_matrix_with_row_names(fullfile(sourceDir, ...
        [ep '_correlation_difference.csv']));
    imagesc(ax,corrDiff);axis(ax,'square','tight');
    colormap(ax,flipud(parula)); caxis(ax,[0 0.5]);
    ax.XTick=1:numel(displayNames);ax.XTickLabel=displayNames;xtickangle(ax,45);
    ax.YTick=1:numel(displayNames);ax.YTickLabel=displayNames;
    apply_supplement_style(ax,s);ax.XGrid='off';ax.YGrid='off';ax.FontSize=7.8;
    panel_label_inside(ax,letters((row-1)*4+3),s);
    cb=colorbar(ax);cb.FontName=s.fontName;cb.FontSize=7.5;
    if row==1
        title(ax,columnTitles{3},'Interpreter','none','FontName',s.fontName, ...
            'FontSize',9.2,'FontWeight','normal');
    end

    % Nearest-neighbor distance comparison.
    ax=nexttile(tl,(row-1)*4+4);hold(ax,'on');
    [dReal,dSynth]=nearest_neighbor_distances(realX,synthX);
    draw_distance_summary(ax,dReal,1,s.blue,s);
    draw_distance_summary(ax,dSynth,2,s.orange,s);
    ax.XTick=[1 2];ax.XTickLabel={'Real-real','Synthetic-real'};
    xtickangle(ax,25);
    ylabel(ax,'Standardized Euclidean distance','FontName',s.fontName, ...
        'FontSize',9.5,'FontWeight','normal');
    xlim(ax,[0.5 2.5]);apply_supplement_style(ax,s);ax.FontSize=8.5;
    panel_label_inside(ax,letters((row-1)*4+4),s);
    if row==1
        title(ax,columnTitles{4},'FontName',s.fontName,'FontSize',10.5, ...
            'FontWeight','normal');
    end
    hold(ax,'off');
end
export_supplement_figure(fig, fullfile(outRoot,'Figure_S10_GaussianCopula_Diagnostics'));
close(fig);
end

%% Figure S11
function make_s11(sourceDir, outRoot, s)
paths={fullfile(sourceDir,'viscosity_full_training_history.csv'), ...
    fullfile(sourceDir,'clearance_full_training_history.csv')};
labels={'Viscosity','Mouse clearance'};
fig=figure('Color','w','Units','centimeters','Position',[2 2 17.5 8.5]);
tl=tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
for k=1:2
    T=readtable(paths{k},'VariableNamingRule','preserve');
    keep=string(T.mode)=="full" & string(T.pooling)=="avg";
    T=T(keep,:);assert(~isempty(T),'No full/avg history rows in %s',paths{k});
    ax=nexttile(tl,k);hold(ax,'on');
    folds=unique(T.fold)';
    for fold=folds
        idx=T.fold==fold;
        [epoch,ord]=sort(T.epoch(idx));
        train=T.loss(idx);train=train(ord);val=T.val_loss(idx);val=val(ord);
        plot(ax,epoch,train,'Color',mix_color(s.trainColor,[1 1 1],0.70), ...
            'LineWidth',0.8,'HandleVisibility','off');
        plot(ax,epoch,val,'Color',mix_color(s.validationColor,[1 1 1],0.70), ...
            'LineWidth',0.8,'HandleVisibility','off');
    end
    epochs=unique(T.epoch);
    meanTrain=nan(size(epochs));meanVal=nan(size(epochs));
    for i=1:numel(epochs)
        idx=T.epoch==epochs(i);
        meanTrain(i)=mean(T.loss(idx),'omitnan');
        meanVal(i)=mean(T.val_loss(idx),'omitnan');
    end
    h1=plot(ax,epochs,meanTrain,'Color',s.trainColor,'LineWidth',s.mainLineWidth, ...
        'DisplayName','Mean training loss');
    h2=plot(ax,epochs,meanVal,'Color',s.validationColor,'LineWidth',s.mainLineWidth, ...
        'DisplayName','Mean validation loss');
    xlabel(ax,'Epoch','FontName',s.fontName,'FontSize',s.labelFontSize, ...
        'FontWeight','normal');
    ylabel(ax,'Log-cosh loss','FontName',s.fontName,'FontSize',s.labelFontSize, ...
        'FontWeight','normal');
    title(ax,labels{k},'FontName',s.fontName,'FontSize',s.labelFontSize, ...
        'FontWeight','normal');
    xlim(ax,[1 max(T.epoch)]);ylim(ax,[0 max([T.loss;T.val_loss])*1.05]);
    apply_supplement_style(ax,s);panel_label(ax,char('a'+k-1),s);
    legend(ax,[h1 h2],{'Mean training loss','Mean validation loss'}, ...
        'Location','northeast','Box','off','FontName',s.fontName, ...
        'FontSize',s.legendFontSize);
    hold(ax,'off');
end
export_supplement_figure(fig, fullfile(outRoot,'Figure_S11_Training_Validation_Loss'));
close(fig);
end

%% Shared helpers
function packageRoot=locate_package_root(startDir)
envRoot=getenv('ACET_PACKAGE_ROOT');
if ~isempty(envRoot) && exist(fullfile(envRoot,'MANIFEST.csv'),'file')==2 && ...
        exist(fullfile(envRoot,'Data'),'dir')==7
    packageRoot=envRoot;
    return;
end
p=startDir;
while true
    if exist(fullfile(p,'MANIFEST.csv'),'file')==2 && exist(fullfile(p,'Data'),'dir')==7
        packageRoot=p;
        return;
    end
    parent=fileparts(p);
    if isempty(parent) || strcmp(parent,p); break; end
    p=parent;
end
error(['Could not locate the ACeT package root. Pass packageRoot or set ' ...
    'ACET_PACKAGE_ROOT. The root must contain MANIFEST.csv and Data/.']);
end

function annotation_text(ax,str,s)
text(ax,0.04,0.96,str,'Units','normalized','VerticalAlignment','top', ...
    'HorizontalAlignment','left','FontName',s.fontName, ...
    'FontSize',s.annotationFontSize,'BackgroundColor','w', ...
    'EdgeColor',[0.75 0.75 0.75],'Margin',5);
end

function annotation_text_at(ax,str,position,s)
text(ax,position(1),position(2),str,'Units','normalized', ...
    'VerticalAlignment','top','HorizontalAlignment','left', ...
    'FontName',s.fontName,'FontSize',s.annotationFontSize, ...
    'BackgroundColor','w','EdgeColor',[0.75 0.75 0.75],'Margin',5);
end

function annotation_text_at_right(ax,str,position,s)
text(ax,position(1),position(2),str,'Units','normalized', ...
    'VerticalAlignment','top','HorizontalAlignment','right', ...
    'FontName',s.fontName,'FontSize',s.annotationFontSize-0.5, ...
    'BackgroundColor','w','EdgeColor',[0.75 0.75 0.75],'Margin',4, ...
    'Clipping','on');
end

function figure_textbox(fig,position,str,s)
annotation(fig,'textbox',position,'String',str,'Interpreter','tex', ...
    'VerticalAlignment','middle','HorizontalAlignment','left', ...
    'FontName',s.fontName,'FontSize',s.annotationFontSize, ...
    'BackgroundColor','w','EdgeColor',[0.75 0.75 0.75], ...
    'LineWidth',0.7,'Margin',6,'FitBoxToText','off');
end

function panel_label(ax,label,s)
text(ax,-0.13,1.06,label,'Units','normalized','FontName',s.fontName, ...
    'FontSize',s.panelFontSize,'FontWeight','bold', ...
    'HorizontalAlignment','left','VerticalAlignment','bottom', ...
    'Clipping','off');
end

function panel_label_inside(ax,label,s)
text(ax,0.015,0.985,label,'Units','normalized','FontName',s.fontName, ...
    'FontSize',s.panelFontSize,'FontWeight','bold', ...
    'HorizontalAlignment','left','VerticalAlignment','top', ...
    'BackgroundColor','w','Margin',1,'Clipping','on');
end

function q=percentile_linear(x,p)
x=sort(x(:));
if isempty(x);q=nan;return;end
pos=1+(numel(x)-1)*p/100;
lo=floor(pos);hi=ceil(pos);
if lo==hi;q=x(lo);else;q=x(lo)+(pos-lo)*(x(hi)-x(lo));end
end

function show_image_panel(ax,path,label,s)
img=imread(path);
img=crop_white_margins(img,12);
image(ax,img);axis(ax,'image','off');
text(ax,0.01,0.99,label,'Units','normalized','FontName',s.fontName, ...
    'FontSize',s.panelFontSize,'FontWeight','bold', ...
    'VerticalAlignment','top','BackgroundColor','w','Margin',2);
end

function img=crop_white_margins(img,pad)
if size(img,3)==4; img=img(:,:,1:3); end
mask=any(img<248,3);
rows=find(any(mask,2)); cols=find(any(mask,1));
if isempty(rows)||isempty(cols); return; end
r1=max(1,rows(1)-pad);r2=min(size(img,1),rows(end)+pad);
c1=max(1,cols(1)-pad);c2=min(size(img,2),cols(end)+pad);
img=img(r1:r2,c1:c2,:);
end

function M=regression_metrics(y,p)
y=y(:);p=p(:);res=y-p;
M.r2=1-sum(res.^2)/sum((y-mean(y)).^2);
r=corr(y,p,'Rows','complete');M.pearson2=r.^2;
M.spearman=corr(y,p,'Type','Spearman','Rows','complete');
M.rmse=sqrt(mean(res.^2));M.mae=mean(abs(res));
end

function [ba,mcc]=triage_metrics(y,p,thr)
cm=confusion_at_threshold(y,p,thr);
tn=cm(1,1);fp=cm(1,2);fn=cm(2,1);tp=cm(2,2);
tnr=tn/(tn+fp);tpr=tp/(tp+fn);ba=(tnr+tpr)/2;
den=sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));
if den==0;mcc=nan;else;mcc=(tp*tn-fp*fn)/den;end
end

function cm=confusion_at_threshold(y,p,thr)
truePoor=y(:)>thr;predPoor=p(:)>thr;
cm=[sum(~truePoor & ~predPoor),sum(~truePoor & predPoor); ...
    sum(truePoor & ~predPoor),sum(truePoor & predPoor)];
end

function limits=parity_limits(y,p,pad)
lo=min([y(:);p(:)]);hi=max([y(:);p(:)]);
limits=[floor(lo-pad),ceil(hi+pad)];
end

function c=contrast_text_color(value,maxValue)
if value/maxValue>0.55;c='k';else;c='w';end
end

function [X,source,names]=diagnostic_matrix(T)
source=string(T.source);
exclude=["endpoint","source","row_id"];
vars=string(T.Properties.VariableNames);
keep=~ismember(vars,exclude);
numKeep=false(size(keep));
for i=1:numel(vars)
    if keep(i)
        numKeep(i)=isnumeric(T.(char(vars(i))));
    end
end
names=cellstr(vars(numKeep));
X=table2array(T(:,names));
end

function labels=short_feature_names(endpoint,names)
if strcmp(endpoint,'viscosity')
    labels={'kD','Plates','AC-SINS','FWHM','Viscosity'};
else
    labels={'Heparin RT','Heparin %B','BVP','poly-D-lysine','AUCt'};
end
if numel(labels)~=numel(names)
    labels=names;
end
end

function label=endpoint_label(ep)
if strcmp(ep,'viscosity');label='Viscosity';else;label='Mouse clearance';end
end

function [scores,explained]=pca_from_combined_scale(realX,synthX)
combined=[realX;synthX];
mu=mean(combined,1);sd=std(combined,0,1);sd(sd==0)=1;
Z=(combined-mu)./sd;
[~,S,V]=svd(Z,'econ');scores=Z*V;
latent=diag(S).^2;explained=latent./sum(latent).*100;
end

function M=read_numeric_matrix_with_row_names(path)
try
    T=readtable(path,'ReadRowNames',true,'VariableNamingRule','preserve');
    M=table2array(T);
catch
    T=readtable(path,'VariableNamingRule','preserve');
    M=table2array(T(:,2:end));
end
M=double(M);
end

function [realDist,synthDist]=nearest_neighbor_distances(realX,synthX)
mu=mean(realX,1);sd=std(realX,0,1);sd(sd==0)=1;
R=(realX-mu)./sd;S=(synthX-mu)./sd;
DR=euclidean_matrix(R,R);DR(1:size(DR,1)+1:end)=inf;
realDist=min(DR,[],2);
DS=euclidean_matrix(S,R);synthDist=min(DS,[],2);
end

function D=euclidean_matrix(A,B)
D2=max(sum(A.^2,2)+sum(B.^2,2)'-2*(A*B'),0);
D=sqrt(D2);
end

function draw_distance_summary(ax,values,x,color,s)
q1=percentile_linear(values,25);q3=percentile_linear(values,75);
med=median(values);
plot(ax,[x x],[q1 q3],'-','Color',color,'LineWidth',5);
plot(ax,[x-0.12 x+0.12],[med med],'-','Color',s.black,'LineWidth',1.5);
rng(100+x);j=(rand(size(values))-0.5)*0.20;
scatter(ax,x+j,values,18,'o','MarkerFaceColor',color,'MarkerEdgeColor','none', ...
    'MarkerFaceAlpha',0.50);
end

function c=mix_color(a,b,f)
c=(1-f).*a+f.*b;
end

function write_source_manifest(packageRoot,outRoot,visSplit,clrSplit,visFixed,clrFixed,clinicalS5Dir,hicAssaysPath,hicPatchPath,viscosityDataPath,synthDir,poolDir)
items={ ...
'S1','Viscosity fixed-holdout seed stability plus composite-split error sensitivity', ...
    sprintf('%s | %s',visFixed,visSplit); ...
'S2','Viscosity fixed-holdout and composite-split parity', ...
    sprintf('%s | %s',visFixed,visSplit); ...
'S3','Mouse-exposure fixed-holdout seed stability plus composite-split error sensitivity', ...
    sprintf('%s | %s',clrFixed,clrSplit); ...
'S4','Mouse-exposure fixed-holdout and composite-split parity', ...
    sprintf('%s | %s',clrFixed,clrSplit); ...
'S5','Clinical external robustness',clinicalS5Dir; ...
'S6','HIC assays-only OOF parity',hicAssaysPath; ...
'S7','HIC patch-only comparison',hicPatchPath; ...
'S8','HIC 30-min triage',sprintf('%s | %s',hicAssaysPath,hicPatchPath); ...
'S9','SEC peak shape vs viscosity',viscosityDataPath; ...
'S10','Gaussian-Copula diagnostics',synthDir; ...
'S11','Training/validation loss curves',poolDir};
T=cell2table(items,'VariableNames',{'Figure','Description','AuthoritativeSource'});
for i=1:height(T)
    T.AuthoritativeSource{i}=package_relative_text(T.AuthoritativeSource{i},packageRoot);
end
writetable(T,fullfile(outRoot,'SUPPLEMENTARY_FIGURE_SOURCE_MANIFEST.csv'));
end

function text=package_relative_text(text,packageRoot)
text=strrep(text,[packageRoot filesep],'');
text=strrep(text,'\','/');
end

function write_output_checksums(outRoot)
D=dir(fullfile(outRoot,'**','*'));
fid=fopen(fullfile(outRoot,'CHECKSUMS.sha256'),'w');
if fid<0;warning('Could not create CHECKSUMS.sha256');return;end
cleanup=onCleanup(@() fclose_if_open(fid));
for i=1:numel(D)
    if D(i).isdir || strcmp(D(i).name,'CHECKSUMS.sha256') || ...
            strcmp(D(i).name,'FIGURE_GENERATION_LOG.txt');continue;end
    path=fullfile(D(i).folder,D(i).name);
    try
        hash=sha256_java(path);
        rel=strrep(path,[outRoot filesep],'');
        fprintf(fid,'%s  %s\n',hash,strrep(rel,'\','/'));
    catch err
        warning('Checksum skipped for %s: %s',path,err.message);
    end
end
clear cleanup;fclose_if_open(fid);
end

function hex=sha256_java(path)
md=java.security.MessageDigest.getInstance('SHA-256');
fid=fopen(path,'rb');assert(fid>=0,'Cannot open %s',path);
cleanup=onCleanup(@() fclose_if_open(fid));
while true
    bytes=fread(fid,1024*1024,'*uint8');
    if isempty(bytes);break;end
    md.update(bytes);
end
raw=typecast(md.digest(),'uint8');
hex=lower(reshape(dec2hex(raw,2).',1,[]));
clear cleanup;fclose_if_open(fid);
end

function log_line(fid,fmt,varargin)
line=sprintf(fmt,varargin{:});fprintf('%s\n',line);
if fid>=0;fprintf(fid,'%s\n',line);end
end

function fclose_if_open(fid)
if ~isempty(fid) && isnumeric(fid) && fid>0
    try fclose(fid);catch,end
end
end
