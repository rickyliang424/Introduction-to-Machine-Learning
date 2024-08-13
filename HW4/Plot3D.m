clear; clc;
load LDA.mat;
load KMeans.mat;

figure('visible','on');
subplot(1,2,1);
plot3(LDA_3_feature(1:1725,1),LDA_3_feature(1:1725,2),LDA_3_feature(1:1725,3),'.',...
      LDA_3_feature(1726:3456,1),LDA_3_feature(1726:3456,2),LDA_3_feature(1726:3456,3),'.',...
      LDA_3_feature(3457:4531,1),LDA_3_feature(3457:4531,2),LDA_3_feature(3457:4531,3),'.',...
      LDA_3_feature(4532:6340,1),LDA_3_feature(4532:6340,2),LDA_3_feature(4532:6340,3),'.','MarkerSize',16);
grid on; set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5);
legend(["C","P","R","I"],'fontsize',12,'fontweight','bold','Location','northeast');
title('Label by gourp');
view(120,50);
% C(1):1-1725
% P(2):1726-3456
% R(3):3457-4531
% I(4):4532-6340

subplot(1,2,2);
plot3(KM_ftr(1392:2078,1),KM_ftr(1392:2078,2),KM_ftr(1392:2078,3),'.',...
      KM_ftr(1:1391,1),KM_ftr(1:1391,2),KM_ftr(1:1391,3),'.',...
      KM_ftr(3515:6340,1),KM_ftr(3515:6340,2),KM_ftr(3515:6340,3),'.',...
      KM_ftr(2079:3514,1),KM_ftr(2079:3514,2),KM_ftr(2079:3514,3),'.','MarkerSize',16);
grid on; set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5);
legend(["0","1","2","3"],'fontsize',12,'fontweight','bold','Location','northeast');
title('Label by clustering');
view(120,50);
% (0):1-1391
% (1):1392-2078
% (2):2079-3514
% (3):3515-6340
