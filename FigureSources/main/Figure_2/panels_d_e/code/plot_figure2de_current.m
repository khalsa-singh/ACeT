function plot_figure2de_current(packageRoot, requestedOutputDir)
%% plot_figure2de_current.m
% Current-authoritative panel producers for manuscript Figure 2d and 2e.
%
% This script preserves the original separate-panel workflow:
%   Fig2D_FeatureImportance_current.*  -> manuscript panel 2d
%   Fig2E_LearningAblation_current.*   -> manuscript panel 2e
%
% It reads only the current CSV outputs produced by
% viscosity_figure2de_current.py. It does not train a model and does not
% alter the archived input data.

close all; clc;

%% Locate project and data
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end

repoRoot = scriptDir;
while ~isfolder(fullfile(repoRoot, 'Data', 'curated'))
    parentDir = fileparts(repoRoot);
    if strcmp(parentDir, repoRoot), error('Cannot locate the ACeT package root.'); end
    repoRoot = parentDir;
end
if nargin >= 1 && ~isempty(packageRoot); repoRoot=char(packageRoot); end
dataDir = fullfile(repoRoot, 'MainPack', 'viscosity', 'results', 'figure2de');
outputDir = fullfile(repoRoot, 'reruns', 'figure2de');
if nargin >= 2 && ~isempty(requestedOutputDir); outputDir=char(requestedOutputDir); end
if ~isfolder(outputDir), mkdir(outputDir); end

requiredFiles = {
    'figure2d_importance_summary.csv'
    'figure2d_seed_importance.csv'
    'figure2d_pairwise_tests.csv'
    'figure2e_ablation_summary.csv'
    'figure2e_ablation_runs.csv'
    'figure2e_ablation_tests.csv'
    'figure2e_learning_curve_summary.csv'
};
for k = 1:numel(requiredFiles)
    thisPath = fullfile(dataDir, requiredFiles{k});
    if ~isfile(thisPath)
        error('Required current result is missing: %s', thisPath);
    end
end

%% ------------------------------------------------------------------------
% Figure 2d: permutation feature importance
% Bars: across-seed means
% Error bars: SD across five training seeds
% Open circles: seed-level means (each seed itself averages 10 permutations)
% Brackets: paired two-sided t-tests across seeds, Holm-Sidak adjusted
% -------------------------------------------------------------------------
Tsum = readtable(fullfile(dataDir, 'figure2d_importance_summary.csv'), ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
Tseed = readtable(fullfile(dataDir, 'figure2d_seed_importance.csv'), ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
Tpair = readtable(fullfile(dataDir, 'figure2d_pairwise_tests.csv'), ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');

% The Python summary is already sorted by decreasing across-seed mean.
features = string(Tsum.feature);
means = double(Tsum.mean_importance);
sds = double(Tsum.sd_across_seeds);
nFeat = height(Tsum);

labels = strings(nFeat,1);
for i = 1:nFeat
    switch features(i)
        case "DLS Interaction Parameter kD (mL/g)"
            labels(i) = "DLS k_D";
        case "SE-UHPLC Main Peak Plates (EP)"
            labels(i) = "SE-UHPLC Plates";
        case "AC-SINS lambda max (nm)"
            labels(i) = "AC-SINS Delta lambda_{max}";
        case "AC-SINS λmax (nm)"
            labels(i) = "AC-SINS \Delta\lambda_{max}";
        case "SE-UHPLC Main Peak FWHM (min)"
            labels(i) = "SE-UHPLC FWHM";
        otherwise
            labels(i) = string(Tsum.feature_display(i));
    end
end

figD = figure('Color','w','Position',[100 100 760 550], ...
    'PaperPositionMode','auto','InvertHardcopy','off');
axD = axes('Parent',figD,'Position',[0.11 0.18 0.86 0.72]);
hold(axD,'on');

barColors = flipud(parula(nFeat));
for i = 1:nFeat
    bar(axD, i, means(i), 0.60, ...
        'FaceColor',barColors(i,:), ...
        'EdgeColor','none');
end

hErr = errorbar(axD, 1:nFeat, means, sds, ...
    'k','LineStyle','none','LineWidth',1.5);
hErr.CapSize = 12;

% Seed-level points, deterministically jittered only for display.
rng(20260808,'twister');
for i = 1:nFeat
    mask = Tseed.feature == features(i);
    vals = double(Tseed.importance_mean(mask));
    xj = i + (rand(size(vals))-0.5)*0.10;
    scatter(axD, xj, vals, 50, ...
        'MarkerEdgeColor','k', ...
        'MarkerFaceColor','none', ...
        'LineWidth',0.9);
end

set(axD, ...
    'XTick',1:nFeat, ...
    'XTickLabel',cellstr(labels), ...
    'TickLabelInterpreter','tex', ...
    'TickDir','out', ...
    'Box','off', ...
    'LineWidth',1.5, ...
    'FontName','Helvetica', ...
    'FontSize',14);
ylabel(axD,'Importance score (decrease in held-out R^2)', ...
    'FontName','Helvetica','FontSize',16,'FontWeight','bold');

% Draw all adjusted pairwise brackets. Heights are separated by the first
% feature index, matching the original panel logic while using current data.
allTops = means + sds;
dataSpan = max(allTops) - min([0; means-sds]);
if dataSpan <= 0
    dataSpan = 1;
end
baseStep = 0.085 * dataSpan;
currentLevel = zeros(nFeat,1);

for k = 1:height(Tpair)
    f1 = string(Tpair.feature_1(k));
    f2 = string(Tpair.feature_2(k));
    i = find(features == f1, 1);
    j = find(features == f2, 1);
    if isempty(i) || isempty(j)
        continue;
    end
    if i > j
        tmp = i; i = j; j = tmp;
    end
    currentLevel(i) = currentLevel(i) + 1;
    yb = max(allTops(i:j)) + currentLevel(i)*baseStep;
    tick = 0.12*baseStep;
    plot(axD,[i i j j],[yb-tick yb yb yb-tick], ...
        'k','LineWidth',1.25);
    pAdj = double(Tpair.holm_sidak_p(k));
    if pAdj < 0.001
        pText = 'p < 0.001';
    else
        pText = sprintf('p = %.3f',pAdj);
    end
    text(axD,(i+j)/2,yb+0.12*baseStep,pText, ...
        'HorizontalAlignment','center', ...
        'FontName','Helvetica','FontSize',11, ...
        'FontAngle','italic');
end

lowerLim = min([0; means-sds]) - 0.08*dataSpan;
upperLim = max([allTops; axD.YLim(2)]) + 4.8*baseStep;
ylim(axD,[lowerLim upperLim]);
xlim(axD,[0.45 nFeat+0.55]);
hold(axD,'off');
drawnow;

print(figD,fullfile(outputDir,'Fig2D_FeatureImportance_current.png'), ...
    '-dpng','-r600');
print(figD,fullfile(outputDir,'Fig2D_FeatureImportance_current.svg'), ...
    '-dsvg','-vector');
print(figD,fullfile(outputDir,'Fig2D_FeatureImportance_current.eps'), ...
    '-depsc2','-vector','-loose');
savefig(figD,fullfile(outputDir,'Fig2D_FeatureImportance_current.fig'));

%% ------------------------------------------------------------------------
% Figure 2e: learning curve and prespecified feature ablations
% -------------------------------------------------------------------------
Tlearn = readtable(fullfile(dataDir, 'figure2e_learning_curve_summary.csv'), ...
    'TextType','string','VariableNamingRule','preserve');
Tabl = readtable(fullfile(dataDir, 'figure2e_ablation_summary.csv'), ...
    'TextType','string','VariableNamingRule','preserve');
Truns = readtable(fullfile(dataDir, 'figure2e_ablation_runs.csv'), ...
    'TextType','string','VariableNamingRule','preserve');
Ttests = readtable(fullfile(dataDir, 'figure2e_ablation_tests.csv'), ...
    'TextType','string','VariableNamingRule','preserve');

% Enforce the prespecified display order.
conditionOrder = ["full","top2","ht_assays","no_kd"];
labelsAbl = {'Full','Top2 (k_D, Plates)','HT assays','No k_D'};
meansAbl = nan(1,4);
sdsAbl = nan(1,4);
for i = 1:4
    idx = find(string(Tabl.condition) == conditionOrder(i),1);
    if isempty(idx)
        error('Missing ablation condition: %s',conditionOrder(i));
    end
    meansAbl(i) = double(Tabl.mean_test_r2(idx));
    sdsAbl(i) = double(Tabl.sd_test_r2(idx));
end

figE = figure('Color','w','Position',[100 100 1200 500], ...
    'PaperPositionMode','auto','InvertHardcopy','off');

% Left: nested learning curve
axL = subplot(1,2,1,'Parent',figE);
hold(axL,'on');
xLearn = double(Tlearn.percent_of_total_75);
yLearn = double(Tlearn.mean_test_r2);
sdLearn = double(Tlearn.sd_test_r2);
if any(sdLearn > 0)
    errorbar(axL,xLearn,yLearn,sdLearn,'-o', ...
        'LineWidth',1.5,'MarkerSize',9, ...
        'Color',[0 0.447 0.741], ...
        'MarkerFaceColor','w','CapSize',8);
else
    plot(axL,xLearn,yLearn,'-o', ...
        'LineWidth',1.5,'MarkerSize',9, ...
        'Color',[0 0.447 0.741], ...
        'MarkerFaceColor','w');
end
set(axL,'Box','off','LineWidth',1.5,'FontSize',16, ...
    'FontName','Helvetica','TickDir','out');
xlabel(axL,'Training set size (% of total dataset)', ...
    'FontName','Helvetica','FontSize',15,'FontWeight','bold');
ylabel(axL,'Held-out test R^2', ...
    'FontName','Helvetica','FontSize',15,'FontWeight','bold');
grid(axL,'on');
axL.GridAlpha = 0.25;
axis(axL,'square');
xlim(axL,[min(xLearn)-2 max(xLearn)+1]);
yMargin = max(0.05,0.08*(max(yLearn)-min(yLearn)+eps));
ylim(axL,[min(-0.05,min(yLearn-sdLearn)-yMargin) ...
          max(0.80,max(yLearn+sdLearn)+yMargin)]);
hold(axL,'off');

% Right: ablation means, SDs and seed-level runs
axR = subplot(1,2,2,'Parent',figE);
hold(axR,'on');
ypos = 1:4;
barColors = lines(4);
for i = 1:4
    barh(axR,i,meansAbl(i),0.60, ...
        'FaceColor',barColors(i,:), ...
        'EdgeColor','none');
end
hErrA = errorbar(axR,meansAbl,ypos,sdsAbl,sdsAbl,'horizontal', ...
    'LineStyle','none','Color','k','LineWidth',1.5);
hErrA.CapSize = 12;

rng(20260808,'twister');
allRunValues = [];
for i = 1:4
    mask = string(Truns.condition) == conditionOrder(i);
    vals = double(Truns.r2(mask));
    allRunValues = [allRunValues; vals(:)]; %#ok<AGROW>
    yj = i + (rand(size(vals))-0.5)*0.14;
    scatter(axR,vals,yj,50, ...
        'MarkerEdgeColor','k','MarkerFaceColor','none','LineWidth',0.9);
end

set(axR,'YTick',ypos,'YTickLabel',labelsAbl,'YDir','reverse', ...
    'TickLabelInterpreter','tex','TickDir','out','Box','off', ...
    'LineWidth',1.5,'FontSize',15,'FontName','Helvetica');
xlabel(axR,'Held-out test R^2', ...
    'FontName','Helvetica','FontSize',15,'FontWeight','bold');

xMinData = min([allRunValues; (meansAbl-sdsAbl)']);
xMaxData = max([allRunValues; (meansAbl+sdsAbl)']);
xSpan = max(0.2,xMaxData-xMinData);
xlim(axR,[min(0,xMinData-0.10*xSpan), xMaxData+0.22*xSpan]);

% Stars report Holm-adjusted paired tests versus the full condition.
for k = 1:height(Ttests)
    cond = string(Ttests.condition(k));
    idx = find(conditionOrder == cond,1);
    if isempty(idx)
        continue;
    end
    pAdj = double(Ttests.holm_p(k));
    if pAdj < 0.001
        star = '***';
    elseif pAdj < 0.01
        star = '**';
    elseif pAdj < 0.05
        star = '*';
    else
        star = 'ns';
    end
    xStar = max([meansAbl(idx)+sdsAbl(idx), ...
        max(double(Truns.r2(string(Truns.condition)==cond)))]) + 0.035*xSpan;
    text(axR,xStar,idx,star, ...
        'HorizontalAlignment','left','VerticalAlignment','middle', ...
        'FontName','Helvetica','FontSize',14,'FontWeight','bold');
end
grid(axR,'off');
axis(axR,'square');
hold(axR,'off');
drawnow;

print(figE,fullfile(outputDir,'Fig2E_LearningAblation_current.png'), ...
    '-dpng','-r600');
print(figE,fullfile(outputDir,'Fig2E_LearningAblation_current.svg'), ...
    '-dsvg','-vector');
print(figE,fullfile(outputDir,'Fig2E_LearningAblation_current.eps'), ...
    '-depsc2','-vector','-loose');
savefig(figE,fullfile(outputDir,'Fig2E_LearningAblation_current.fig'));

fprintf('Created current Figure 2d/2e panels in:\n%s\n',outputDir);
