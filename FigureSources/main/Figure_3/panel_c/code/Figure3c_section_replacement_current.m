%% Figure 3c: current-authoritative clearance baseline comparison
% No training. Bars are held-out metrics; diamonds/fold points are current CV arrays.
data=load('figure3c_current_baselines.mat'); models={'ACeT','Ridge','SVR','Random forest'};
barColors=[.20 .60 .80;.75 .85 .95;.75 .85 .95;.75 .85 .95]; rng(0); jitter=(rand(5,4)-.5)*.15;
figure('Color','w','Units','normalized','Position',[.05 .10 .90 .70]);
subplot(2,2,[1 2]); hold on;
for i=1:4, bar(i,data.test_r2_all(i),'FaceColor',barColors(i,:),'BarWidth',.5,'EdgeColor','none'); end
scatter(1:4,mean(data.cv_r2,1),100,'d','MarkerEdgeColor',[.3 .3 .3],'MarkerFaceColor',[.3 .3 .3]);
for i=1:4, scatter(i+jitter(:,i),data.cv_r2(:,i),80,'o','MarkerEdgeColor',[.2 .2 .2],'MarkerFaceColor','none','LineWidth',1); end
minR2=min([data.cv_r2(:);data.test_r2_all(:)]); ylim([min(-0.05,floor((minR2-.02)*10)/10) 1]); ylabel('R^2'); xticks(1:4); xticklabels(models);
set(gca,'FontSize',24,'FontName','Helvetica','LineWidth',1.5,'TickDir','out','Box','off');
subplot(2,2,3); hold on;
for i=1:4, bar(i,data.test_rmse_all(i),'FaceColor',barColors(i,:),'BarWidth',.5,'EdgeColor','none'); end
scatter(1:4,mean(data.cv_rmse,1),100,'d','MarkerEdgeColor',[.3 .3 .3],'MarkerFaceColor',[.3 .3 .3]);
for i=1:4, scatter(i+jitter(:,i),data.cv_rmse(:,i),80,'o','MarkerEdgeColor',[.2 .2 .2],'MarkerFaceColor','none','LineWidth',1); end
ylabel('RMSE'); xticks(1:4); xticklabels(models); xtickangle(45); set(gca,'FontSize',24,'FontName','Helvetica','LineWidth',1.5,'TickDir','out','Box','off');
subplot(2,2,4); hold on;
for i=1:4, bar(i,data.test_mae_all(i),'FaceColor',barColors(i,:),'BarWidth',.5,'EdgeColor','none'); end
scatter(1:4,mean(data.cv_mae,1),100,'d','MarkerEdgeColor',[.3 .3 .3],'MarkerFaceColor',[.3 .3 .3]);
for i=1:4, scatter(i+jitter(:,i),data.cv_mae(:,i),80,'o','MarkerEdgeColor',[.2 .2 .2],'MarkerFaceColor','none','LineWidth',1); end
ylabel('MAE'); xticks(1:4); xticklabels(models); xtickangle(45); set(gca,'FontSize',24,'FontName','Helvetica','LineWidth',1.5,'TickDir','out','Box','off');
