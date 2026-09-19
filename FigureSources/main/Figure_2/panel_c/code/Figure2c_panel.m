%% ————— Figure 2c: corrected CV-fold and held-out comparator metrics —————
% Replace the previous model-comparison section with this block.
% Required input: figure2c_corrected_baselines.mat

data = load('figure2c_corrected_baselines.mat');

cv_r2   = data.cv_r2;
cv_rmse = data.cv_rmse;
cv_mae  = data.cv_mae;

test_r2   = data.test_r2_all(:)';
test_rmse = data.test_rmse_all(:)';
test_mae  = data.test_mae_all(:)';

models = {'ACeT','Ridge','SVR','Random forest'};

barColors = [
    0.20 0.60 0.80;
    0.75 0.85 0.95;
    0.75 0.85 0.95;
    0.75 0.85 0.95
];
diamondColor = [0.30 0.30 0.30];
circleEdgeColor = [0.20 0.20 0.20];

rng(0);
jitter = (rand(5,4)-0.5)*0.15;

figure('Color','w','Units','normalized','Position',[0.05 0.10 0.90 0.70]);

subplot(2,2,[1,2]);
hold on;
for i=1:4
    bar(i,test_r2(i),'FaceColor',barColors(i,:), ...
        'BarWidth',0.5,'EdgeColor','none');
end
scatter(1:4,mean(cv_r2,1),100,'d', ...
    'MarkerEdgeColor',diamondColor,'MarkerFaceColor',diamondColor);
for i=1:4
    scatter(i+jitter(:,i),cv_r2(:,i),80,'o', ...
        'MarkerEdgeColor',circleEdgeColor,'MarkerFaceColor','none','LineWidth',1);
end
ylim([0 1]);
ylabel('R^2','FontSize',24);
xticks(1:4);
xticklabels(models);
set(gca,'FontSize',24,'FontName','Helvetica','LineWidth',1.5, ...
    'TickDir','out','Box','off');
hBars=findobj(gca,'Type','Bar');
hMean=findobj(gca,'Type','Scatter','Marker','d');
hFold=findobj(gca,'Type','Scatter','Marker','o');
legend([hBars(end),hMean(1),hFold(end)], ...
    {'Test set','CV mean','CV folds'}, ...
    'Location','best','FontSize',20,'Box','off');

subplot(2,2,3);
hold on;
for i=1:4
    bar(i,test_rmse(i),'FaceColor',barColors(i,:), ...
        'BarWidth',0.5,'EdgeColor','none');
end
scatter(1:4,mean(cv_rmse,1),100,'d', ...
    'MarkerEdgeColor',diamondColor,'MarkerFaceColor',diamondColor);
for i=1:4
    scatter(i+jitter(:,i),cv_rmse(:,i),80,'o', ...
        'MarkerEdgeColor',circleEdgeColor,'MarkerFaceColor','none','LineWidth',1);
end
ylabel('RMSE (cP)','FontSize',24);
xticks(1:4);
xticklabels(models);
xtickangle(45);
ylim([0,1.15*max([test_rmse(:);cv_rmse(:)])]);
set(gca,'FontSize',24,'FontName','Helvetica','LineWidth',1.5, ...
    'TickDir','out','Box','off');

subplot(2,2,4);
hold on;
for i=1:4
    bar(i,test_mae(i),'FaceColor',barColors(i,:), ...
        'BarWidth',0.5,'EdgeColor','none');
end
scatter(1:4,mean(cv_mae,1),100,'d', ...
    'MarkerEdgeColor',diamondColor,'MarkerFaceColor',diamondColor);
for i=1:4
    scatter(i+jitter(:,i),cv_mae(:,i),80,'o', ...
        'MarkerEdgeColor',circleEdgeColor,'MarkerFaceColor','none','LineWidth',1);
end
ylabel('MAE (cP)','FontSize',24);
xticks(1:4);
xticklabels(models);
xtickangle(45);
ylim([0,1.15*max([test_mae(:);cv_mae(:)])]);
set(gca,'FontSize',24,'FontName','Helvetica','LineWidth',1.5, ...
    'TickDir','out','Box','off');

print(gcf,'Figure_2c_corrected.png','-dpng','-r600');
saveas(gcf,'Figure_2c_corrected.svg');
print(gcf,'Figure_2c_corrected.eps','-depsc2','-vector');
