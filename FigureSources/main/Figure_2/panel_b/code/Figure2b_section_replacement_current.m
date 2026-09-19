%% Figure 2b: current-authoritative fixed viscosity ranges
% No training. Input values come only from the packaged current prediction CSV.
data = load('figure2b_current_fixed_ranges.mat');
labels = cellstr(data.range_labels); rmseMean = data.bootstrap_rmse_mean(:);
figure('Color','w','Units','pixels','Position',[100 100 760 650]); hold on; grid on;
plot([0 50],[0 50],'k--','LineWidth',1.5);
errorbar(data.true_means(:),data.pred_means(:),rmseMean,'o','MarkerSize',12, ...
 'MarkerFaceColor',[0.20 0.70 0.90],'MarkerEdgeColor','k','LineWidth',2, ...
 'Color',[0 0.45 0.74],'CapSize',15,'LineStyle','none');
rng(double(data.bootstrap_seed));
for i=1:3
 s=data.bootstrap_rmse_samples(:,i); jitter=(rand(size(s))-0.5)*0.10;
 scatter(data.true_means(i)+jitter,data.pred_means(i)+s,10,'k','filled','MarkerFaceAlpha',0.05,'MarkerEdgeAlpha',0.05);
 scatter(data.true_means(i)+jitter,data.pred_means(i)-s,10,'k','filled','MarkerFaceAlpha',0.05,'MarkerEdgeAlpha',0.05);
 text(data.true_means(i)+1.4,data.pred_means(i),sprintf('%s (n=%d)',labels{i},data.bin_counts(i)), ...
  'FontSize',16,'FontWeight','bold','HorizontalAlignment','left');
end
axis square; xlim([0 50]); ylim([0 50]); xlabel('Mean true viscosity (cP)'); ylabel('Mean predicted viscosity (cP)');
set(gca,'FontSize',16,'FontName','Helvetica','LineWidth',1.5,'TickDir','out','Box','off');
