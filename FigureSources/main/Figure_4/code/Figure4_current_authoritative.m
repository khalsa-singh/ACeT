function Figure4_current_authoritative(packageRoot, outputDir)
% Assemble Figure 4 from the archived clinical predictions, thresholds and economics.
if nargin < 1 || isempty(packageRoot)
    packageRoot = fileparts(mfilename('fullpath'));
    while ~exist(fullfile(packageRoot,'Data','curated'),'dir')
        parent = fileparts(packageRoot);
        assert(~strcmp(parent,packageRoot),'ACeT root not found.');
        packageRoot = parent;
    end
end
if nargin < 2 || isempty(outputDir); outputDir = fullfile(packageRoot,'reruns','figure4'); end
if ~exist(outputDir,'dir'); mkdir(outputDir); end
packageDir = outputDir;
clinicalDir = fullfile(packageRoot,'MainPack','clinical_outcome','results','targeted_repro');
matPath = fullfile(clinicalDir,'clinical_figure_data.mat');
thresholdPath = fullfile(clinicalDir,'ASSAY_THRESHOLD_COMPARISON.csv');
economicsPath = fullfile(clinicalDir,'FIGURE4_ECONOMICS_RECALC.csv');
rulePath = fullfile(packageRoot,'MainPack','clinical_outcome','results','finalization','TABLE_S14_CLINICAL_RULE_COMPARISON.csv');

S = load(matPath,'cm_train','cm_test','cm_external','class_names');
Tthr = readtable(thresholdPath,'VariableNamingRule','preserve');
Tecon = readtable(economicsPath,'VariableNamingRule','preserve');
Trules = readtable(rulePath,'VariableNamingRule','preserve'); %#ok<NASGU>

% Preserve the historical Approved -> Terminated ordering.
classes = string(S.class_names);
ord = [find(contains(classes,'Approved','IgnoreCase',true)), ...
       find(contains(classes,'Terminated','IgnoreCase',true))];
labels = classes(ord);
cmTrain = double(S.cm_train(ord,ord));
cmTest = double(S.cm_test(ord,ord));
cmExternal = double(S.cm_external(ord,ord));

% Current threshold values; red thresholds are independent raw-assay
% univariate comparators selected on the 89-antibody training cohort.
features = string(Tthr.feature);
displayFeatures = replace(features,"ACSINS","AC-SINS");
publishedThresholds = Tthr.jain_rule_threshold;
trainingThresholds = Tthr.threshold;
absoluteDeltas = abs(Tthr.training_minus_jain_delta);

% Locked economics, selected by explicit basis and policy.
getNet = @(basis,policy) Tecon.net_B(strcmp(string(Tecon.basis),basis) & strcmp(string(Tecon.policy),policy));
bars = [getNet("internal_prior_expected_N23","ACeT"), ...
        getNet("internal_prior_expected_N23","develop_all"), ...
        getNet("internal_prior_expected_N23","kill_all"), ...
        getNet("external_realized_2_approved_12_terminated","ACeT"), ...
        getNet("external_realized_2_approved_12_terminated","develop_all"), ...
        getNet("external_realized_2_approved_12_terminated","kill_all")];
assert(max(abs(bars-[6.639512615384618,-15.640,-23.13984,10.672,-6.000,-16.992])) < 1e-10);

% Economic cell values and expected internal counts preserve the panel 4c source.
econCell = [9.0,-9.0;-2.0,0.084];
PoS = 0.12; N = sum(cmTest(:));
sens = cmTest(1,1)/sum(cmTest(1,:)); spec = cmTest(2,2)/sum(cmTest(2,:));
winners = N*PoS; losers = N-winners;
expectedCount = [winners*sens,winners*(1-sens);losers*(1-spec),losers*spec];
expectedUSD = expectedCount.*econCell;
realizedUSD = cmExternal.*econCell;

%% Composite layout
fig = figure('Color','w','Units','centimeters','Position',[1 1 34 25], ...
    'Renderer','painters');

% Panel a: reference bubble geometry, percentages, colors, and labels.
axA1 = axes(fig,'Position',[0.075 0.585 0.165 0.325]);
drawBubbleCM(axA1,cmTrain,labels,'Internal training');
axA2 = axes(fig,'Position',[0.295 0.585 0.165 0.325]);
drawBubbleCM(axA2,cmTest,labels,'Internal held-out test');
ylabel(axA2,'');
annotation(fig,'textbox',[0.018 0.935 0.03 0.03],'String','a','EdgeColor','none', ...
    'FontName','Helvetica','FontSize',18,'FontWeight','bold');

% Panel b: same grouped horizontal bars, white/red styling, and deltas.
axB = axes(fig,'Position',[0.555 0.585 0.40 0.325]); hold(axB,'on');
C = categorical(displayFeatures,displayFeatures,displayFeatures);
bh = barh(axB,C,[publishedThresholds,trainingThresholds],'BarWidth',0.7);
bh(1).FaceColor=[1 1 1]; bh(1).EdgeColor='k';
bh(2).FaceColor=[1 0 0]; bh(2).EdgeColor='k';
axB.YDir='reverse';
for k=1:numel(features)
    x1=publishedThresholds(k); x2=trainingThresholds(k);
    plot(axB,[x1 x2],[k k],'k-','LineWidth',1.2);
    text(axB,max(x1,x2)+0.02*max([publishedThresholds;trainingThresholds]),k, ...
        sprintf('\\Delta = %.2f',absoluteDeltas(k)),'FontName','Helvetica', ...
        'FontSize',10,'FontWeight','bold','Clipping','off');
end
legend(axB,{'Published single-assay','Training-optimized single-assay'}, ...
    'Location','southeast','FontName','Helvetica','FontSize',10,'Box','off');
xlabel(axB,'Threshold (assay-specific units)','FontName','Helvetica','FontSize',13,'FontWeight','bold');
ylabel(axB,'Assay','FontName','Helvetica','FontSize',13,'FontWeight','bold');
title(axB,'Single-assay threshold comparators for ACeT','FontName','Helvetica','FontSize',14,'FontWeight','bold');
styleAxes(axB); grid(axB,'on'); axB.GridColor=[0.85 0.85 0.85]; axB.GridAlpha=0.3;
axB.XLim=[0 13.5];
annotation(fig,'textbox',[0.515 0.935 0.03 0.03],'String','b','EdgeColor','none', ...
    'FontName','Helvetica','FontSize',18,'FontWeight','bold');

% Panel c: preserve expected-internal and realized-external economic matrices.
vmax=max(abs([expectedUSD(:);realizedUSD(:)]));
cmap=[ones(32,1),linspace(0,1,32)',linspace(0,1,32)'; ...
      linspace(1,0,32)',ones(32,1),linspace(1,0,32)'];
axC1=axes(fig,'Position',[0.075 0.105 0.165 0.325]);
drawEconomicMatrix(axC1,expectedUSD,expectedCount,labels,'Expected internal portfolio',vmax,cmap,false);
axC2=axes(fig,'Position',[0.295 0.105 0.165 0.325]);
drawEconomicMatrix(axC2,realizedUSD,cmExternal,labels,'Realized external cohort',vmax,cmap,true);
cb=colorbar(axC2,'Position',[0.470 0.145 0.010 0.245]); cb.Label.String='Impact (B$)'; cb.FontSize=9;
annotation(fig,'textbox',[0.018 0.455 0.03 0.03],'String','c','EdgeColor','none', ...
    'FontName','Helvetica','FontSize',18,'FontWeight','bold');

% Panel d: locked values and corrected policy labels.
axD=axes(fig,'Position',[0.575 0.105 0.38 0.325]);
colors=[0.00 0.45 0.74;0.85 0.33 0.10;0.65 0.05 0.05; ...
        0.30 0.75 0.93;0.99 0.55 0.38;0.80 0.30 0.30];
hold(axD,'on');
for k=1:6, bar(axD,k,bars(k),'FaceColor',colors(k,:)); end
hold(axD,'off');
axD.XTick=1:6;
axD.XTickLabel={'Int ACeT','Int Dev-all','Int Kill-all','Ext ACeT','Ext Dev-all','Ext Kill-all'};
xtickangle(axD,25); axD.YLim=[-27 15];
ylabel(axD,'Net Impact (B$)','FontName','Helvetica','FontSize',13,'FontWeight','bold');
for k=1:6
    dy=1.5*sign(bars(k));
    text(axD,k,bars(k)+dy,sprintf('%+.1f',bars(k)),'HorizontalAlignment','center', ...
        'FontName','Helvetica','FontSize',12,'FontWeight','bold');
end
styleAxes(axD); grid(axD,'on');
annotation(fig,'textbox',[0.515 0.455 0.03 0.03],'String','d','EdgeColor','none', ...
    'FontName','Helvetica','FontSize',18,'FontWeight','bold');

%% Save complete source data and required exports.
source = struct();
source.cm_train_in_sample_ensemble=cmTrain;
source.cm_internal_test=cmTest;
source.cm_external_realized=cmExternal;
source.class_names=cellstr(labels);
source.features=cellstr(features);
source.published_thresholds=publishedThresholds;
source.training_optimized_univariate_thresholds=trainingThresholds;
source.absolute_threshold_deltas=absoluteDeltas;
source.expected_internal_counts=expectedCount;
source.expected_internal_USD_B=expectedUSD;
source.realized_external_USD_B=realizedUSD;
source.policy_net_USD_B=bars;
source.policy_labels={'Int ACeT','Int Dev-all','Int Kill-all','Ext ACeT','Ext Dev-all','Ext Kill-all'};
save(fullfile(packageDir,'Figure4_FINAL_source.mat'),'-struct','source');

values = table( ...
    ["4a";"4a";"4a";"4a";"4a";"4a";"4a";"4a";"4d";"4d";"4d";"4d";"4d";"4d"], ...
    ["training_in_sample_ensemble";"training_in_sample_ensemble";"training_in_sample_ensemble";"training_in_sample_ensemble"; ...
     "internal_test";"internal_test";"internal_test";"internal_test"; ...
     "internal_prior_expected_N23";"internal_prior_expected_N23";"internal_prior_expected_N23"; ...
     "external_realized_2_approved_12_terminated";"external_realized_2_approved_12_terminated";"external_realized_2_approved_12_terminated"], ...
    ["TP";"FN";"FP";"TN";"TP";"FN";"FP";"TN";"ACeT_net_B";"develop_all_net_B";"kill_all_net_B";"ACeT_net_B";"develop_all_net_B";"kill_all_net_B"], ...
    [cmTrain(1,1);cmTrain(1,2);cmTrain(2,1);cmTrain(2,2);cmTest(1,1);cmTest(1,2);cmTest(2,1);cmTest(2,2);bars(:)], ...
    'VariableNames',{'panel','cohort_or_basis','quantity','value'});
writetable(values,fullfile(packageDir,'Figure4_FINAL_source_values.csv'));

exportgraphics(fig,fullfile(packageDir,'Figure4_FINAL_current.png'),'Resolution',600);
exportgraphics(fig,fullfile(packageDir,'Figure4_FINAL_current.pdf'),'ContentType','vector');
print(fig,fullfile(packageDir,'Figure4_FINAL_current.svg'),'-dsvg','-vector');
print(fig,fullfile(packageDir,'Figure4_FINAL_current.eps'),'-depsc2','-vector','-loose');
rgb=imread(fullfile(packageDir,'Figure4_FINAL_current.png'));
if size(rgb,3)==4, rgb=rgb(:,:,1:3); end
imwrite(rgb,fullfile(packageDir,'Figure4_FINAL_current_RGB.tif'),'tif','Compression','lzw','Resolution',600);

end

function drawBubbleCM(ax,cm,labels,ttl)
    pct=cm./sum(cm,2)*100; [X,Y]=meshgrid(1:2,1:2);
    areas=(sqrt(cm(:)./max(cm(:)))*sqrt(12000)).^2;
    areas(cm(:)>0 & areas<3000)=3000;
    scatter(ax,X(:),Y(:),areas,pct(:),'o','filled','MarkerEdgeColor','k', ...
        'MarkerFaceAlpha',0.8,'LineWidth',0.8);
    for k=1:4
        text(ax,X(k),Y(k),sprintf('%d\n(%d%%)',cm(k),round(pct(k))), ...
            'HorizontalAlignment','center','VerticalAlignment','middle', ...
            'FontName','Helvetica','FontSize',11,'FontWeight','bold');
    end
    colormap(ax,'parula'); caxis(ax,[floor(min(pct(:))) ceil(max(pct(:)))]);
    ax.YDir='reverse'; ax.XLim=[0.5 2.5]; ax.YLim=[0.5 2.5];
    ax.XTick=1:2; ax.YTick=1:2; ax.XTickLabel=labels; ax.YTickLabel=labels;
    xtickangle(ax,35); title(ax,ttl,'FontSize',12,'FontWeight','bold');
    xlabel(ax,'Predicted Outcome','FontWeight','bold'); ylabel(ax,'True Outcome','FontWeight','bold');
    styleAxes(ax);
end

function drawEconomicMatrix(ax,usd,counts,labels,ttl,vmax,cmap,isInteger)
    imagesc(ax,usd); axis(ax,'equal','tight'); ax.YDir='reverse';
    ax.XTick=1:2; ax.YTick=1:2; ax.XTickLabel=labels; ax.YTickLabel=labels;
    colormap(ax,cmap); caxis(ax,[-vmax vmax]); names={'TP','FN';'FP','TN'};
    for i=1:2
        for j=1:2
            if isInteger, countText=sprintf('%d',counts(i,j)); else, countText=sprintf('%.2f',counts(i,j)); end
            displayUSD=usd(i,j);
            if abs(displayUSD)<0.005
                usdText='0.00 B$';
            else
                usdText=sprintf('%+.2f B$',displayUSD);
            end
            text(ax,j,i,sprintf('%s  %s\n%s',names{i,j},countText,usdText), ...
                'HorizontalAlignment','center','FontName','Helvetica','FontSize',10,'FontWeight','bold');
        end
    end
    title(ax,ttl,'FontSize',12,'FontWeight','bold'); styleAxes(ax);
end

function styleAxes(ax)
    ax.FontName='Helvetica'; ax.FontSize=10; ax.LineWidth=0.75; ax.TickDir='out'; ax.Box='off';
end
