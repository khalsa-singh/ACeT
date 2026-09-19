function plot_mouse_exposure_panels(packageRoot, requestedOutputDir)
% Mouse-exposure figure producers
% Updated panel-by-panel producer for manuscript Figure 3.
%
% This script preserves the original separate-panel workflow:
%   Figure 3a: current-authoritative clearance parity plot and error CDF
%   Figure 3b: retained matched augmented-versus-unaugmented experiment
%   Figure 3c: current-authoritative comparator panel
%   Figure 3d: retained SHAP analysis associated with the augmented run in 3b
%
% No model training is performed in MATLAB.



close all;
clc;

%% ------------------------------------------------------------------------
% Paths
% -------------------------------------------------------------------------
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
outputDir = fullfile(repoRoot, 'reruns', 'figure3');
if nargin >= 2 && ~isempty(requestedOutputDir); outputDir=char(requestedOutputDir); end
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
panel3APath = fullfile(repoRoot, 'MainPack', 'mouse_exposure', 'results', 'primary', 'clearance_panel3A_data.mat');
panel3CPath = fullfile(repoRoot, 'FigureSources', 'main', 'Figure_3', 'panel_c', 'data', 'figure3c_current_baselines.mat');
shapPath = fullfile(repoRoot, 'MainPack', 'mouse_exposure', 'results', 'figure3d', 'clearance_shap_summary.mat');

fprintf('Figure 3a source: %s\n', panel3APath);
fprintf('Figure 3c source: %s\n', panel3CPath);
fprintf('Figure 3d source: %s\n', shapPath);
fprintf('Outputs: %s\n\n', outputDir);

%% ========================================================================
% Figure 3a: current-authoritative parity plot in the ORIGINAL layout/style
% ========================================================================
S3a = load(panel3APath);
trueStored = getFirstField(S3a, {'trues','y_true','y_test','true_values'});
predStored = getFirstField(S3a, {'preds','y_pred','predictions','pred_values'});

trueStored = double(trueStored(:));
predStored = double(predStored(:));
valid = isfinite(trueStored) & isfinite(predStored) & trueStored ~= 0;
trueStored = trueStored(valid);
predStored = predStored(valid);

assert(numel(trueStored) == 11, ...
    'Figure 3a should contain exactly 11 held-out antibodies.');
assert(numel(predStored) == numel(trueStored), ...
    'Figure 3a true and predicted vectors must have equal length.');

r2A = coefficientOfDetermination(trueStored, predStored);
rmseA = sqrt(mean((trueStored - predStored).^2));
nrmseA = rmseA / mean(trueStored);
maeA = mean(abs(trueStored - predStored));

assert(abs(r2A - 0.810019264) < 5e-4, ...
    'Figure 3a R^2 does not match the current-authoritative anchor.');
assert(abs(nrmseA - 0.143225995) < 5e-4, ...
    'Figure 3a nRMSE does not match the current-authoritative anchor.');

% The current MAT stores AUC after division by 10^4. Restore the original
% ng-h/mL display scale, while retaining the reference panel layout.
assert(max(trueStored) < 1e5, ...
    ['Figure 3a target values do not look like the expected stored scale. ' ...
     'Stop rather than applying 10^4 twice.']);
aucDisplayScale = 1e4;
xA = trueStored * aucDisplayScale;
yA = predStored * aucDisplayScale;

% Density estimation on log10 scale, as in the original script.
maskA = xA > 0 & yA > 0;
xxA = xA(maskA);
yyA = yA(maskA);
ptsA = [log10(xxA(:)), log10(yyA(:))];
nPointA = size(ptsA,1);
sigmaA = std(ptsA,1);
bandwidthA = 1.06 .* sigmaA .* (nPointA^(-1/5));
bandwidthA(bandwidthA <= 0 | ~isfinite(bandwidthA)) = eps;
densityA = mvksdensity(ptsA, ptsA, 'Bandwidth', bandwidthA);

% Original plot-bound calculation.
dataValuesA = [xA(:); yA(:)];
minA = min(dataValuesA);
maxA = max(dataValuesA);
rangeA = maxA - minA;
bufferA = (rangeA == 0) * max(abs(minA),1) * 0.1 + ...
          (rangeA ~= 0) * rangeA * 0.1;
hiA = maxA + bufferA;
if hiA <= minA
    hiA = maxA + 1;
end

% ORIGINAL 600 x 600 figure, default axes placement, labels and inset.
figA = figure('Color','w','Position',[100 100 600 600], ...
    'PaperPositionMode','auto','InvertHardcopy','off');
axA = axes('Parent',figA);
scatter(axA, xA, yA, 50, densityA, 'filled', ...
    'MarkerEdgeColor','none');
hold(axA,'on');
plot(axA, [0 hiA], [0 hiA], '--', ...
    'Color',[0.4 0.4 0.4], 'LineWidth',1);
hold(axA,'off');
axis(axA, [0 hiA 0 hiA]);
grid(axA,'on');
set(axA, ...
    'FontName','Helvetica', ...
    'FontSize',12, ...
    'LineWidth',1.2, ...
    'Box','off', ...
    'TickDir','out', ...
    'GridLineStyle',':', ...
    'GridColor',[0.8 0.8 0.8]);

colormap(axA, parula);
cbA = colorbar(axA,'EastOutside');
cbA.Label.String = 'Point Density (log scale)';
cbA.Label.FontSize = 12;

% Preserve the exact concise labels used in the original Figure 3a.
xlabel(axA, 'WT mouse \it{AUCt_{e}} \rm{(ng-h/mL)}', ...
    'FontSize',18,'FontWeight','bold');
ylabel(axA, 'WT mouse \it{AUCt_{p}} \rm{(ng-h/mL)}', ...
    'FontSize',18,'FontWeight','bold');

% Original top-left metric box position and size.
metricTextA = {sprintf('R^2 = %.2f',r2A), ...
               sprintf('nRMSE = %.2f',nrmseA)};
annotation(figA,'textbox',[0.15 0.90 0.01 0.01], ...
    'String',metricTextA, ...
    'FitBoxToText','on', ...
    'BackgroundColor','w', ...
    'EdgeColor',[0.6 0.6 0.6], ...
    'FontSize',16, ...
    'FontName','Helvetica', ...
    'LineWidth',0.8);

% Original southeast CDF inset placement and dimensions.
pctErrorA = abs((yA - xA) ./ xA) * 100;
sortedPctA = sort(pctErrorA);
numPctA = numel(sortedPctA);
cumPctA = (1:numPctA)' ./ numPctA * 100;

axInsetA = axes('Parent',figA,'Position',[0.55 0.20 0.25 0.25]);
plot(axInsetA,sortedPctA,cumPctA,'r-','LineWidth',1.8);
hold(axInsetA,'on');
percentileLevelsA = [50 75 90];
percentileValuesA = zeros(size(percentileLevelsA));
for qIndex = 1:numel(percentileLevelsA)
    qLevel = percentileLevelsA(qIndex);
    idxQ = max(1,round(numPctA*qLevel/100));
    qValue = sortedPctA(idxQ);
    percentileValuesA(qIndex) = qValue;
    xline(axInsetA,qValue,'--', ...
        'Color',[0.5 0.5 0.5],'LineWidth',1.2);
    text(axInsetA,qValue,qLevel+3, ...
        sprintf('%d%% = %.1f%%',qLevel,qValue), ...
        'Rotation',90, ...
        'FontSize',9, ...
        'Color',[0.5 0.5 0.5], ...
        'VerticalAlignment','bottom');
end
hold(axInsetA,'off');

xlim(axInsetA,[0 50]);
set(axInsetA, ...
    'FontName','Helvetica', ...
    'FontSize',10, ...
    'LineWidth',1, ...
    'Box','off', ...
    'TickDir','out');
ylabel(axInsetA,'% Samples','FontSize',10,'FontWeight','bold');
xlabel(axInsetA,'% Error','FontSize',10,'FontWeight','bold');

savePanel(figA,outputDir,'Figure_3a_clearance_parity_with_inset',600);
savePanel(figA,outputDir,'clearance_parity_with_inset',600);

fprintf(['Figure 3a: R2=%.6f, RMSE=%.6f stored units, ' ...
    'nRMSE=%.6f, MAE=%.6f; CDF 50/75/90=%.2f/%.2f/%.2f%%\n'], ...
    r2A,rmseA,nrmseA,maeA,percentileValuesA(1), ...
    percentileValuesA(2),percentileValuesA(3));

%% ========================================================================
% Figure 3b: retained augmented-versus-unaugmented experiment,
%            using the ORIGINAL size, concise labels and panel layout
% ========================================================================
obsB = [7.393e6, 10.033e6, 2.721e6, 7.523e6, 7.185e6, ...
        8.693e6, 5.650e6, 4.691e6, 8.266e6, 8.194e6, 12.405e6];
predAugB = [6.955e6, 9.524e6, 3.815e6, 6.939e6, 6.137e6, ...
            8.647e6, 7.994e6, 6.832e6, 8.120e6, 8.134e6, 11.779e6];
predNoAugB = [6.972e6, 9.665e6, 3.968e6, 6.969e6, 8.096e6, ...
              9.595e6, 9.606e6, 7.017e6, 8.194e6, 8.356e6, 9.649e6];

% Correct coefficient of determination; do not use squared correlation.
r2AugB = coefficientOfDetermination(obsB,predAugB);
r2NoAugB = coefficientOfDetermination(obsB,predNoAugB);
assert(abs(r2AugB - 0.7979254) < 1e-6, ...
    'Figure 3b augmented R^2 does not match the retained paired analysis.');
assert(abs(r2NoAugB - 0.5164624) < 1e-6, ...
    'Figure 3b unaugmented R^2 does not match the retained paired analysis.');

% Exact original 600 x 600 panel geometry.
figB = figure('Color','w','Position',[100 100 600 600], ...
    'PaperPositionMode','auto','InvertHardcopy','off');
axB = axes('Parent',figB);
hold(axB,'on');

lineWidthB = 1.5;
markerSizeB = 80;
scatter(axB,obsB,predAugB,markerSizeB,'o', ...
    'MarkerFaceColor',[1 0 0], ...
    'MarkerEdgeColor','k', ...
    'LineWidth',lineWidthB, ...
    'DisplayName',sprintf('Augmented (R^2=%.2f)',r2AugB));
scatter(axB,obsB,predNoAugB,markerSizeB,'s', ...
    'MarkerFaceColor','none', ...
    'MarkerEdgeColor','k', ...
    'LineWidth',lineWidthB, ...
    'DisplayName',sprintf('No augmentation (R^2=%.2f)',r2NoAugB));

limsB = [0,max([obsB,predAugB,predNoAugB])*1.05];
arrowFractionB = 0.8;
arrowHeadLengthB = diff(limsB)*0.02;

% Preserve the original convention: arrows only where augmentation moves
% the prediction closer to the observed value.
for i = 1:numel(obsB)
    noAugError = abs(predNoAugB(i)-obsB(i));
    augError = abs(predAugB(i)-obsB(i));
    if augError <= noAugError
        dx = 0;
        dy = predAugB(i)-predNoAugB(i);
        xEnd = obsB(i)+arrowFractionB*dx;
        yEnd = predNoAugB(i)+arrowFractionB*dy;
        plot(axB,[obsB(i),xEnd],[predNoAugB(i),yEnd],'--', ...
            'Color',[0.3 0.3 0.3], ...
            'LineWidth',lineWidthB, ...
            'HandleVisibility','off');
        thetaB = atan2(dy,dx);
        headX = [xEnd, ...
            xEnd-arrowHeadLengthB*cos(thetaB-pi/6), ...
            xEnd-arrowHeadLengthB*cos(thetaB+pi/6)];
        headY = [yEnd, ...
            yEnd-arrowHeadLengthB*sin(thetaB-pi/6), ...
            yEnd-arrowHeadLengthB*sin(thetaB+pi/6)];
        patch(axB,headX,headY,[0.3 0.3 0.3], ...
            'EdgeColor','none','HandleVisibility','off');
    end
end

plot(axB,limsB,limsB,'k--','LineWidth',lineWidthB, ...
    'DisplayName','y = x');
xlim(axB,limsB);
ylim(axB,limsB);
axis(axB,'square');

% Preserve the exact concise axis labels from the original panel.
xlabel(axB,'WT mouse \it{AUCt_{e}} \rm{(ng-h/mL)}', ...
    'FontSize',18,'FontWeight','bold');
ylabel(axB,'WT mouse \it{AUCt_{p}} \rm{(ng-h/mL)}', ...
    'FontSize',18,'FontWeight','bold');
set(axB, ...
    'FontName','Helvetica', ...
    'FontSize',14, ...
    'LineWidth',1.2, ...
    'TickDir','out');
legend(axB,'show','Location','southeast','Box','off','FontSize',14);
hold(axB,'off');

savePanel(figB,outputDir,'Figure_3b_augmented_vs_unaugmented',600);
savePanel(figB,outputDir,'Fig3D_augmented_vs_unaugmented_linear',600);

fprintf('Figure 3b: augmented R2=%.6f; no-augmentation R2=%.6f\n', ...
    r2AugB,r2NoAugB);

%% ========================================================================
% Figure 3c: current-authoritative CV-versus-held-out comparison
% ========================================================================
S3c = load(panel3CPath);

% Support both current MAT layouts:
%   (1) consolidated 5 x 4 arrays: cv_r2/cv_nrmse/cv_nmae; or
%   (2) one field per model, as written by clearance_results_cv.mat.
if isfield(S3c,'cv_r2') && isfield(S3c,'cv_nrmse') && isfield(S3c,'cv_nmae')
    cvR2 = double(S3c.cv_r2);
    cvNRMSE = double(S3c.cv_nrmse);
    cvNMAE = double(S3c.cv_nmae);
else
    requiredSplitFields = { ...
        'cv_r2_transformer','cv_r2_ridge','cv_r2_svr','cv_r2_rf', ...
        'cv_nrmse_transformer','cv_nrmse_ridge','cv_nrmse_svr','cv_nrmse_rf', ...
        'cv_nmae_transformer','cv_nmae_ridge','cv_nmae_svr','cv_nmae_rf'};
    for fieldIndex = 1:numel(requiredSplitFields)
        assert(isfield(S3c,requiredSplitFields{fieldIndex}), ...
            'Figure 3c MAT is missing field: %s',requiredSplitFields{fieldIndex});
    end
    cvR2 = [ ...
        S3c.cv_r2_transformer(:), ...
        S3c.cv_r2_ridge(:), ...
        S3c.cv_r2_svr(:), ...
        S3c.cv_r2_rf(:)];
    cvNRMSE = [ ...
        S3c.cv_nrmse_transformer(:), ...
        S3c.cv_nrmse_ridge(:), ...
        S3c.cv_nrmse_svr(:), ...
        S3c.cv_nrmse_rf(:)];
    cvNMAE = [ ...
        S3c.cv_nmae_transformer(:), ...
        S3c.cv_nmae_ridge(:), ...
        S3c.cv_nmae_svr(:), ...
        S3c.cv_nmae_rf(:)];
end

assert(isfield(S3c,'test_r2_all') && ...
       isfield(S3c,'test_nrmse_all') && ...
       isfield(S3c,'test_nmae_all'), ...
       'Figure 3c MAT is missing one or more held-out metric arrays.');
testR2 = double(S3c.test_r2_all(:)');
testNRMSE = double(S3c.test_nrmse_all(:)');
testNMAE = double(S3c.test_nmae_all(:)');

assert(isequal(size(cvR2), [5 4]), 'Figure 3c cv_r2 must be 5 x 4.');
assert(isequal(size(cvNRMSE), [5 4]), 'Figure 3c cv_nrmse must be 5 x 4.');
assert(isequal(size(cvNMAE), [5 4]), 'Figure 3c cv_nmae must be 5 x 4.');
assert(numel(testR2) == 4 && numel(testNRMSE) == 4 && numel(testNMAE) == 4, ...
    'Figure 3c held-out metric arrays must each contain four models.');
assert(abs(testR2(1) - 0.791087) < 5e-4, ...
    'Figure 3c ACeT held-out R^2 does not match the current comparator anchor.');

models3c = {'ACeT','Ridge','SVR','Random forest'};
metricNames3c = {'R^2','nRMSE','nMAE'};
markers3c = {'d','s','^'};
cvMeans3c = [mean(cvR2,1); mean(cvNRMSE,1); mean(cvNMAE,1)];
testValues3c = [testR2; testNRMSE; testNMAE];

metricColors3c = [ ...
    0.20 0.60 0.80; ...
    0.85 0.33 0.10; ...
    0.47 0.67 0.19];
lightColors3c = metricColors3c + (1 - metricColors3c) * 0.55;

foldSize3c = 55;
foldSizeACeT3c = 85;
meanSize3c = 135;
meanSizeACeT3c = 220;
foldAlpha3c = 0.45;
lineWidth3c = 1.4;
lineWidthACeT3c = 2.2;

rng(0);
xJitter3c = (rand(5,4,3)-0.5) * 0.012;
minCV3c = min([cvR2(:); cvNRMSE(:); cvNMAE(:)]);
xLower3c = min(-0.05, floor((minCV3c-0.02)*10)/10);

figC = figure('Color','w','Units','normalized', ...
    'Position',[0.18 0.18 0.47 0.70], ...
    'PaperPositionMode','auto','InvertHardcopy','off');
axC = axes('Parent',figC);
hold(axC, 'on');
plot(axC, [0 1], [0 1], 'k--', 'LineWidth',1.2);

legendHandles3c = gobjects(3,1);
for modelIndex = 1:4
    if modelIndex == 1
        foldMarkerSize = foldSizeACeT3c;
        meanMarkerSize = meanSizeACeT3c;
        currentLineWidth = lineWidthACeT3c;
        currentColors = metricColors3c;
        modelFontWeight = 'bold';
    else
        foldMarkerSize = foldSize3c;
        meanMarkerSize = meanSize3c;
        currentLineWidth = lineWidth3c;
        currentColors = lightColors3c;
        modelFontWeight = 'normal';
    end

    for metricIndex = 1:3
        switch metricIndex
            case 1
                cvValues = cvR2(:,modelIndex);
                testValue = testR2(modelIndex);
            case 2
                cvValues = cvNRMSE(:,modelIndex);
                testValue = testNRMSE(modelIndex);
            otherwise
                cvValues = cvNMAE(:,modelIndex);
                testValue = testNMAE(modelIndex);
        end

        scatter(axC, ...
            cvValues + xJitter3c(:,modelIndex,metricIndex), ...
            repmat(testValue,5,1), ...
            foldMarkerSize, ...
            'o', ...
            'MarkerEdgeColor','none', ...
            'MarkerFaceColor',currentColors(metricIndex,:), ...
            'MarkerFaceAlpha',foldAlpha3c);

        metricHandle = scatter(axC, ...
            cvMeans3c(metricIndex,modelIndex), ...
            testValues3c(metricIndex,modelIndex), ...
            meanMarkerSize, ...
            markers3c{metricIndex}, ...
            'MarkerEdgeColor',currentColors(metricIndex,:), ...
            'MarkerFaceColor',currentColors(metricIndex,:), ...
            'LineWidth',currentLineWidth);

        if modelIndex == 1
            legendHandles3c(metricIndex) = metricHandle;
        end
    end

    if modelIndex == 3
        textY3c = testR2(modelIndex) + 0.045;
    else
        textY3c = testR2(modelIndex) + 0.028;
    end
    text(axC, cvMeans3c(1,modelIndex), textY3c, models3c{modelIndex}, ...
        'HorizontalAlignment','center', ...
        'FontName','Helvetica', ...
        'FontSize',17, ...
        'FontWeight',modelFontWeight);
end

xlim(axC, [xLower3c 1.0]);
ylim(axC, [0 1.0]);
axis(axC, 'square');
xlabel(axC, 'Cross-validation value', ...
    'FontName','Helvetica','FontSize',19,'FontWeight','normal');
ylabel(axC, 'Held-out test value', ...
    'FontName','Helvetica','FontSize',19,'FontWeight','normal');
legend(axC, legendHandles3c, metricNames3c, ...
    'Location','southeast', ...
    'Box','off', ...
    'FontSize',14);
set(axC, ...
    'FontName','Helvetica', ...
    'FontSize',17, ...
    'LineWidth',1.2, ...
    'TickDir','out', ...
    'Box','off');
hold(axC, 'off');

savePanel(figC, outputDir, 'Figure_3c_current', 600);

fprintf(['Figure 3c held-out R2: ACeT %.6f, Ridge %.6f, ' ...
    'SVR %.6f, RF %.6f\n'], testR2(1), testR2(2), testR2(3), testR2(4));

%% ========================================================================
% Figure 3d: retained KernelSHAP beeswarm for the augmented run in 3b
% ========================================================================
S3d = load(shapPath);
featureNames3d = string(S3d.feature_names);
shapValues3d = double(S3d.shap_values);
XTest3d = double(S3d.X_test);

assert(size(shapValues3d,1) == 11 && size(shapValues3d,2) == 4, ...
    'Figure 3d SHAP matrix must be 11 x 4.');
assert(isequal(size(XTest3d), size(shapValues3d)), ...
    'Figure 3d X_test and SHAP matrices must have the same dimensions.');

meanAbs3d = mean(abs(shapValues3d), 1);
[meanSorted3d, order3d] = sort(meanAbs3d, 'descend');
shapSorted3d = shapValues3d(:,order3d);
XSorted3d = XTest3d(:,order3d);
featureSorted3d = featureNames3d(order3d);
[nSample3d, nFeature3d] = size(shapSorted3d);

figD = figure('Color','w','Position',[100 100 820 650], ...
    'PaperPositionMode','auto','InvertHardcopy','off');
axD = axes('Parent',figD, 'Position',[0.25 0.13 0.61 0.78]);
hold(axD, 'on');

maxShap3d = max(abs(shapSorted3d(:))) * 1.08;
xlim(axD, [-maxShap3d maxShap3d]);
grid(axD, 'on');
axD.GridColor = [0.8 0.8 0.8];
axD.GridAlpha = 0.3;
axD.YGrid = 'off';
axD.XGrid = 'on';

rng(0);
jitter3d = 0.15;
dotSize3d = 120;
for featureIndex = 1:nFeature3d
    yPosition = nFeature3d - featureIndex + 1;
    xValues = shapSorted3d(:,featureIndex);
    colorValues = XSorted3d(:,featureIndex);
    jitteredY = yPosition + (rand(nSample3d,1)-0.5)*jitter3d;
    scatter(axD, xValues, jitteredY, dotSize3d, colorValues, 'filled', ...
        'MarkerEdgeColor','none');
end

meanY3d = nFeature3d:-1:1;
meanHandle3d = scatter(axD, meanSorted3d, meanY3d, dotSize3d*1.8, 'o', ...
    'MarkerEdgeColor','k', ...
    'MarkerFaceColor','r', ...
    'LineWidth',1.5, ...
    'DisplayName','Mean |SHAP|');

colormap(axD, parula);
cbD = colorbar(axD, 'eastoutside');
cbD.Label.String = 'Feature value (scaled)';
cbD.Label.FontName = 'Helvetica';
cbD.Label.FontSize = 14;
cbD.FontName = 'Helvetica';
cbD.FontSize = 12;
cbD.Box = 'off';

xlabel(axD, 'SHAP value', ...
    'FontName','Helvetica', ...
    'FontSize',18, ...
    'FontWeight','normal');
yticks(axD, 1:nFeature3d);
yticklabels(axD, featureSorted3d(end:-1:1));
axD.YDir = 'normal';
axD.TickDir = 'out';
axD.TickLabelInterpreter = 'none';
axD.FontName = 'Helvetica';
axD.FontSize = 17;
axD.LineWidth = 1.2;
axD.Box = 'off';
ylim(axD, [0.5 nFeature3d+0.5]);
legend(axD, meanHandle3d, ...
    'Location','southeast', ...
    'Box','off', ...
    'FontSize',15);
hold(axD, 'off');

savePanel(figD, outputDir, 'Figure_3d_SHAPsummary_beeswarm', 600);
% Legacy alias retained for the user's existing assembly workflow.
savePanel(figD, outputDir, 'Fig3C_SHAPsummary_beeswarm_improved', 600);

fprintf('Figure 3d ranking: %s\n', strjoin(cellstr(featureSorted3d), ' > '));
fprintf('\nAll Figure 3 panels were written to:\n%s\n', outputDir);

%% ========================================================================
% Local helper functions
% ========================================================================
end

function pathOut = resolveInput(candidates, description)
    pathOut = '';
    for i = 1:numel(candidates)
        candidate = candidates{i};
        if exist(candidate, 'file') == 2
            pathOut = candidate;
            return;
        end
    end

    candidateList = strjoin(candidates, newline);
    error('Unable to locate %s. Checked:%s%s', ...
        description, newline, candidateList);
end

function value = getFirstField(S, possibleNames)
    for i = 1:numel(possibleNames)
        name = possibleNames{i};
        if isfield(S, name)
            value = S.(name);
            return;
        end
    end
    error('None of the expected fields was found: %s', ...
        strjoin(possibleNames, ', '));
end

function r2 = coefficientOfDetermination(yTrue, yPred)
    yTrue = double(yTrue(:));
    yPred = double(yPred(:));
    denominator = sum((yTrue - mean(yTrue)).^2);
    if denominator <= 0
        error('R^2 is undefined because the true values have zero variance.');
    end
    r2 = 1 - sum((yTrue - yPred).^2) / denominator;
end

function savePanel(figHandle, outputDir, fileStem, dpi)
    drawnow;
    set(figHandle, 'PaperPositionMode','auto');
    print(figHandle, fullfile(outputDir, [fileStem '.png']), ...
        '-dpng', sprintf('-r%d', dpi));
    saveas(figHandle, fullfile(outputDir, [fileStem '.svg']));
    print(figHandle, fullfile(outputDir, [fileStem '.eps']), ...
        '-depsc2', '-vector');
    savefig(figHandle, fullfile(outputDir, [fileStem '.fig']));
end
