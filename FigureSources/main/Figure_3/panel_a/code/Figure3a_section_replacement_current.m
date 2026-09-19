%% Figure 3a: current-authoritative clearance parity and error CDF
% No training. All plotted arrays are loaded from the current-authoritative MAT.
data=load('figure3a_current_parity_cdf.mat');
figure('Color','w','Units','normalized','Position',[0.10 0.15 0.80 0.55]);
subplot(1,2,1); scatter(data.trues,data.preds,80,data.dens,'filled','MarkerEdgeColor','k'); hold on;
lims=[min([data.trues(:);data.preds(:)]) max([data.trues(:);data.preds(:)])]; plot(lims,lims,'k--','LineWidth',1.5); axis square;
xlabel('Measured clearance'); ylabel('Predicted clearance'); title(sprintf('R^2 = %.3f',data.r2)); colorbar;
set(gca,'FontSize',16,'FontName','Helvetica','LineWidth',1.5,'TickDir','out','Box','off');
subplot(1,2,2); plot(data.sorted_pct,data.cum_percent,'o-','Color',[0.20 0.60 0.80],'MarkerFaceColor',[0.20 0.60 0.80],'LineWidth',2);
xlabel('Absolute percentage error (%)'); ylabel('Cumulative observations (%)'); ylim([0 100]); grid on;
set(gca,'FontSize',16,'FontName','Helvetica','LineWidth',1.5,'TickDir','out','Box','off');
