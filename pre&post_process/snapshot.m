function  snapshot(shape_att_sla,sla_huatu,x2,y2,u,v,zz,cc,tt,pp)
% This function is used to plot the eddy detection results for a particular day
%   shape_att_sla is the shape of the eddies detected by the model.
%   sla_huatu is SLA arrays.
%   x2 and y2 are the latitude and longitude of the SLA distribution.
%   u and v are geostrophic Velocity.
%   zz is the time dimension of eddy detection results.
%   cc is colormat.
%   tt is the title.
%   pp is the coordinate axis output category.

dd =4;  % Velocity vector arrow interval
d = 0.4; % Velocity vector arrow size
m_proj('equidistant','lon',[173 192.875],'lat',[52 61.875]);
hold on
h2 = m_pcolor(x2,y2,squeeze(sla_huatu(zz,:,:)));
set(h2, 'LineStyle','none');
hold on
lons = []; lats = [];
for j = 1:length(shape_att_sla(zz).lonlat)
    lons = shape_att_sla(zz).lonlat{1,j}(1,:);
    lons = smooth(lons,3);
    lats = shape_att_sla(zz).lonlat{1,j}(2,:);
    lats = smooth(lats,3);
    if shape_att_sla(zz).type(j) == 1

        m_plot(lons,lats,'Color',[0.019608000,0.18823500,0.38039199],'Marker','.','Linewidth',1.5,'MarkerSize',4);
        hold on
    else
        m_plot(lons,lats,'Color',[102,36,0]/255,'Marker','.','Linewidth',1.5,'MarkerSize',4);
        hold on
    end
end
    
colormap(cc)
caxis([-0.1 0.3]);
m_gshhs_h('patch',[69 85 89]/255,'EdgeColor',[69 85 89]/255);
set(gca, 'XTick', 150:25:209.875);
if pp == 1
m_grid('linest','none','linewid',1.5,'FontSize',10,'FontName','Times New Roman','FontWeight','bold'...
    ,'xtick',[],'ytick',([52 54 56 58 60]),'fontsize',20);
elseif pp == 2
    m_grid('linest','none','linewid',1.5,'FontSize',10,'FontName','Times New Roman','FontWeight','bold'...
    ,'xtick',([175 179 183 187 191]),'ytick',([52 54 56 58 60]),'fontsize',20);
elseif pp ==3
m_grid('linest','none','linewid',1.5,'FontSize',10,'FontName','Times New Roman','FontWeight','bold'...
    ,'xtick',[],'ytick',[],'fontsize',20);
elseif pp ==4
m_grid('linest','none','linewid',1.5,'FontSize',10,'FontName','Times New Roman','FontWeight','bold'...
    ,'xtick',([175 179 183 187 191]),'ytick',[],'fontsize',20);
end
m_quiver(x2(1:dd:end,1:dd:end),y2(1:dd:end,1:dd:end),u(1:dd:end,1:dd:end,zz)./d,v(1:dd:end,1:dd:end,zz)./d,0,'k');The trailing 0 determines whether the arrow scales up or down with the figure
.../is to adjust the size of the arrow
title(tt,'fontsize',20,'FontWeight','bold');
end

