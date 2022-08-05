
clear;clc;
% This script is used to process and analysis the Lagrangian eddy tracking results.
load 'E:\term\eddy_matlab\wenzhanghuatu\track_data_new\eddy_tracks.mat';
%%
lon1=173:0.125:193;
lat1=52:0.125:62;
start_date = datenum(1993,01,01);
end_date = datenum(2020,12,31); 
T = start_date:end_date;
dd= 0.25;
[y2,x2]=meshgrid(lat1(1):dd:lat1(end),lon1(1):dd:lon1(end));
[m,n] = size(x2);
burn_anti_num = zeros(m,n);burn_cyc_num = zeros(m,n);
die_anti_num = zeros(m,n);die_cyc_num = zeros(m,n);
for j = 1:m
    for k = 1:n
        for i = 1:length(tracks)
            if length(tracks(i).day)>28
                    % anti
                    if tracks(i).type(1,1) == -1
                        lons_b = tracks(i).lon(1,1);
                        lats_b = tracks(i).lat(1,1);
                        lons_d = tracks(i).lon(end);
                        lats_d = tracks(i).lat(end);
                        if(lons_b>((j-1)*dd+lon1(1))&&lons_b<(j*dd+lon1(1))&&lats_b>((k-1)*dd+lat1(1))&&lats_b<(k*dd+lat1(1)))
                            burn_anti_num(j,k) = burn_anti_num(j,k)+1;
                        end         
                        if(lons_d>((j-1)*dd+lon1(1))&&lons_d<(j*dd+lon1(1))&&lats_d>((k-1)*dd+lat1(1))&&lats_d<(k*dd+lat1(1)))
                            die_anti_num(j,k) = die_anti_num(j,k)+1;
                        end
                    end
                    % cyc
                    if tracks(i).type(1,1) == 1
                        lons_b = tracks(i).lon(1,1);
                        lats_b = tracks(i).lat(1,1);
                        lons_d = tracks(i).lon(end);
                        lats_d = tracks(i).lat(end);
                        if(lons_b>((j-1)*dd+lon1(1))&&lons_b<(j*dd+lon1(1))&&lats_b>((k-1)*dd+lat1(1))&&lats_b<(k*dd+lat1(1)))
                            burn_cyc_num(j,k) = burn_cyc_num(j,k)+1;
                        end
                        if(lons_d>((j-1)*dd+lon1(1))&&lons_d<(j*dd+lon1(1))&&lats_d>((k-1)*dd+lat1(1))&&lats_d<(k*dd+lat1(1)))
                            die_cyc_num(j,k) = die_cyc_num(j,k)+1;
                        end
                    end
            end
        end
    end
end

%%
figure

set(gcf,'position',[100   100   1920  1200]);
m_proj('equidistant','lon',[173 192.875],'lat',[52 61.875]);
hold on
for i=1:length(tracks)
    lon_cyc = [];lat_cyc = [];lon_anti = [];lat_anti = [];
    ty=tracks(i).type;
    z=ty(1,1);
    if length(tracks(i).day)>28

        if z>0%cyc blue
            lon_cyc(:,1)=tracks(i).lon;
            lat_cyc(:,1)=tracks(i).lat ;
            lon_cyc = smooth(lon_cyc,111);lat_cyc = smooth(lat_cyc,111);
            lon0_cyc=lon_cyc(1,1);
            lon1_cyc=lon_cyc(end,1);
            lat0_cyc=lat_cyc(1,1);
            lat1_cyc=lat_cyc(end,1);
            p0_cyc=m_plot(lon0_cyc,lat0_cyc,'marker','o','markersize',5,'Color',[0.019608000,0.18823500,0.38039199]);%,'LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','b', 'MarkerFaceColor',[0.5,0.5,0.5]);
            p1_cyc=m_plot(lon1_cyc,lat1_cyc,'marker','x','markersize',6,'Color',[0.019608000,0.18823500,0.38039199]);%,'LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','b', 'MarkerFaceColor',[0.5,0.5,0.5]);
            p_cyc=m_plot(lon_cyc,lat_cyc,'Color',[0.019608000,0.18823500,0.38039199],'Linewidth',1.2);
            hold on
        else if z<0%anti red
                lon_anti(:,1)=tracks(i).lon;
                lat_anti(:,1)=tracks(i).lat;
                lon_anti = smooth(lon_anti,111);lat_anti = smooth(lat_anti,111);
                lon0_anti=lon_anti(1,1);
                lon1_anti=lon_anti(end,1);
                lat0_anti=lat_anti(1,1);
                lat1_anti=lat_anti(end,1);
                p0_anti= m_plot(lon0_anti,lat0_anti,'marker','o','markersize',5,'Color',[0.41545600,0.0036909999,0.12341400]);%,'LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','r', 'MarkerFaceColor',[0.5,0.5,0.5]);
                p1_anti=m_plot(lon1_anti,lat1_anti,'marker','x','markersize',6,'Color',[0.41545600,0.0036909999,0.12341400]);%,'LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','r', 'MarkerFaceColor',[0.5,0.5,0.5]);                
                p_anti=m_plot(lon_anti,lat_anti,'Color',[0.41545600,0.0036909999,0.12341400],'Linewidth',1.2);
                hold on
            end
        end
    end
end
hold on
m_gshhs_f('patch',[.9 .9 .9],'edgecolor','k');
m_grid('linest','--','linewid',1.5,'FontSize',20,'FontName','Times New Roman', 'FontWeight','bold')
title('Eddy tracks','fontsize',20,'FontWeight','bold');
set(gcf,'Units','points');
pos = get(gcf,'Position');
set(gcf,'PaperPositionMode','Auto','PaperUnits','points','PaperSize',[pos(3), pos(4)])
filename = 'track_new2.pdf'; 
% print(gcf,filename,'-painters','-dpdf')
exportgraphics(gcf,filename,'ContentType','vector')
close(gcf)

