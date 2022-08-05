clc; clear

% This script is used to complement the output of multiple models with each other

cd 'E:\term\eddy_matlab\wenzhanghuatu\'

% Loading the data from modeltrack results
shape_dan_sla = loadcs('..\track_data_dan_onlysla2\',1);
shape_att_sla = loadcs('..\track_data_att_onlysla3\',1);
shape_en_sla = loadcs('..\track_data_en_onlysla3\',1);
shape_dan_suv = loadcs('..\track_data_dan_suv2\',1);
shape_att_suv = loadcs('..\track_data_att_suv3\',1);
shape_en_suv = loadcs('..\track_data_en_suv3\',1);
shape_vg = loadcs('..\vg_track_data2\',1);
%%
for i = 1:length(shape_att_suv)
dan_suv_ll = shape_dan_suv(i).lonlat; % dan_suv
dan_suv_t = shape_dan_suv(i).type;
dan_suv_i = shape_dan_suv(i).i;
dan_suv_j = shape_dan_suv(i).j;
dan_suv_lon = shape_dan_suv(i).lon;
dan_suv_lat = shape_dan_suv(i).lat;

dan_sla_ll = shape_dan_sla(i).lonlat; % dan_sla
dan_sla_t = shape_dan_sla(i).type;
dan_sla_i = shape_dan_sla(i).i;
dan_sla_j = shape_dan_sla(i).j;
dan_sla_lon = shape_dan_sla(i).lon;
dan_sla_lat = shape_dan_sla(i).lat;


en_suv_ll = shape_en_suv(i).lonlat; % en_suv
en_suv_t = shape_en_suv(i).type;
en_suv_i = shape_en_suv(i).i;
en_suv_j = shape_en_suv(i).j;
en_suv_lon = shape_en_suv(i).lon;
en_suv_lat = shape_en_suv(i).lat;

en_sla_ll = shape_en_sla(i).lonlat; % en_sla
en_sla_t = shape_en_sla(i).type;
en_sla_i = shape_en_sla(i).i;
en_sla_j = shape_en_sla(i).j;
en_sla_lon = shape_en_sla(i).lon;
en_sla_lat = shape_en_sla(i).lat;

att_suv_ll = shape_att_suv(i).lonlat; % att_suv
att_suv_t = shape_att_suv(i).type;
att_suv_i = shape_att_suv(i).i;
att_suv_j = shape_att_suv(i).j;
att_suv_lon = shape_att_suv(i).lon;
att_suv_lat = shape_att_suv(i).lat;

att_sla_ll = shape_att_sla(i).lonlat; % att_sla
att_sla_t = shape_att_sla(i).type;
att_sla_i = shape_att_sla(i).i;
att_sla_j = shape_att_sla(i).j;
att_sla_lon = shape_att_sla(i).lon;
att_sla_lat = shape_att_sla(i).lat;

% Using att_suv as a base result
shape_new(i).type = att_suv_t; % shape_new
shape_new(i).i = att_suv_i;
shape_new(i).j = att_suv_j;
shape_new(i).lon = att_suv_lon;
shape_new(i).lat = att_suv_lat;
shape_new(i).lonlat = att_suv_ll;


for j = 1:length(att_sla_ll) % Select a eddy from other model results
    n =0;
    if  all(inpolygon(att_suv_lon,att_suv_lat,att_sla_ll{1,j}(1,:),att_sla_ll{1,j}(2,:))==0) % Determine if the centre of a vortex is surrounded by that eddy in the base result
        for k =1:length(att_suv_lat)
            if all(inpolygon(att_sla_ll{1,j}(1,:),att_sla_ll{1,j}(2,:),att_suv_ll{1,k}(1,:),att_suv_ll{1,k}(2,:))==0) % Determine if the boundary of the eddy intersects with other eddies
                n = n+1;
            end
        end
    end
    if n == length(att_suv_lat) % n==length(base) means that the eddy is not present in the base model results
        if length(att_sla_ll)>5  
            shape_new(i).type = [shape_new(i).type,att_sla_t(j)]; % Add this eddy to the base model
            shape_new(i).i = [shape_new(i).i,att_sla_i(j)];
            shape_new(i).j = [shape_new(i).j,att_sla_j(j)];
            shape_new(i).lon = [shape_new(i).lon,att_sla_lon(j)];
            shape_new(i).lat = [shape_new(i).lat,att_sla_lat(j)];
            shape_new(i).lonlat = [shape_new(i).lonlat,att_sla_ll{1,j}];
        end
    end
end

for j = 1:length(dan_sla_ll)
    n = 0;
    if all(inpolygon(shape_new(i).lon,shape_new(i).lat,dan_sla_ll{1,j}(1,:),dan_sla_ll{1,j}(2,:))==0) % dan_sla
        
        for k =1:length(shape_new(i).lat)
            if all(inpolygon(dan_sla_ll{1,j}(1,:),dan_sla_ll{1,j}(2,:),shape_new(i).lonlat{1,k}(1,:),shape_new(i).lonlat{1,k}(2,:))==0)
                n = n+1;
            end
        end
        if n == length(shape_new(i).lon)
            if length(dan_sla_ll)>5
                shape_new(i).type = [shape_new(i).type,dan_sla_t(j)];
                shape_new(i).i = [shape_new(i).i,dan_sla_i(j)];
                shape_new(i).j = [shape_new(i).j,dan_sla_j(j)];
                shape_new(i).lon = [shape_new(i).lon,dan_sla_lon(j)];
                shape_new(i).lat = [shape_new(i).lat,dan_sla_lat(j)];
                shape_new(i).lonlat = [shape_new(i).lonlat,dan_sla_ll{1,j}];
            end
        end
    end
end

for j = 1:length(dan_suv_ll)
    n = 0;
    if all(inpolygon(shape_new(i).lon,shape_new(i).lat,dan_suv_ll{1,j}(1,:),dan_suv_ll{1,j}(2,:))==0) % dan_sla
        
        for k =1:length(shape_new(i).lat)
            if all(inpolygon(dan_suv_ll{1,j}(1,:),dan_suv_ll{1,j}(2,:),shape_new(i).lonlat{1,k}(1,:),shape_new(i).lonlat{1,k}(2,:))==0)
                n = n+1;
            end
        end
        if n == length(shape_new(i).lon)
            if length(dan_suv_ll)>5
                shape_new(i).type = [shape_new(i).type,dan_suv_t(j)];
                shape_new(i).i = [shape_new(i).i,dan_suv_i(j)];
                shape_new(i).j = [shape_new(i).j,dan_suv_j(j)];
                shape_new(i).lon = [shape_new(i).lon,dan_suv_lon(j)];
                shape_new(i).lat = [shape_new(i).lat,dan_suv_lat(j)];
                shape_new(i).lonlat = [shape_new(i).lonlat,dan_suv_ll{1,j}];
            end
        end
    end
end

for j = 1:length(en_suv_ll)
    n = 0;
    if all(inpolygon(shape_new(i).lon,shape_new(i).lat,en_suv_ll{1,j}(1,:),en_suv_ll{1,j}(2,:))==0) % dan_sla
        
        for k =1:length(shape_new(i).lat)
            if all(inpolygon(en_suv_ll{1,j}(1,:),en_suv_ll{1,j}(2,:),shape_new(i).lonlat{1,k}(1,:),shape_new(i).lonlat{1,k}(2,:))==0)
                n = n+1;
            end
        end
        if n == length(shape_new(i).lon)
            if length(en_suv_ll)>5
                shape_new(i).type = [shape_new(i).type,en_suv_t(j)];
                shape_new(i).i = [shape_new(i).i,en_suv_i(j)];
                shape_new(i).j = [shape_new(i).j,en_suv_j(j)];
                shape_new(i).lon = [shape_new(i).lon,en_suv_lon(j)];
                shape_new(i).lat = [shape_new(i).lat,en_suv_lat(j)];
                shape_new(i).lonlat = [shape_new(i).lonlat,en_suv_ll{1,j}];
            end
        end
    end
end

for j = 1:length(en_sla_ll)
    n = 0;
    if all(inpolygon(shape_new(i).lon,shape_new(i).lat,en_sla_ll{1,j}(1,:),en_sla_ll{1,j}(2,:))==0) % dan_sla
        
        for k =1:length(shape_new(i).lat)
            if all(inpolygon(en_sla_ll{1,j}(1,:),en_sla_ll{1,j}(2,:),shape_new(i).lonlat{1,k}(1,:),shape_new(i).lonlat{1,k}(2,:))==0)
                n = n+1;
            end
        end
        if n == length(shape_new(i).lon)
            if length(en_sla_ll)>5
                shape_new(i).type = [shape_new(i).type,en_sla_t(j)];
                shape_new(i).i = [shape_new(i).i,en_sla_i(j)];
                shape_new(i).j = [shape_new(i).j,en_sla_j(j)];
                shape_new(i).lon = [shape_new(i).lon,en_sla_lon(j)];
                shape_new(i).lat = [shape_new(i).lat,en_sla_lat(j)];
                shape_new(i).lonlat = [shape_new(i).lonlat,en_sla_ll{1,j}];
            end
        end
    end
end

end
%% Counting the number of vortices per day
vg_num = [];att_suv_num = [];dan_suv_num = [];en_suv_num = [];new_num = [];
for i = 1:length(shape_vg)
vg_num(i) = length(shape_vg(i).lonlat);
att_suv_num(i) = length(shape_att_suv(i).lonlat);
dan_suv_num(i) = length(shape_dan_suv(i).lonlat);
en_suv_num(i) = length(shape_en_suv(i).lonlat);
new_num(i) = length(shape_new(i).lonlat);
end
%% load sla and uv
sla = ncread('E:\term\mesoscale_eddy\sla_concat\eddy_sla6.nc','sla');
sla_huatu = sla(6575:end,:,:);
ssu = ncread('..\ssu_bsc.nc','ssu');
u = ssu(:,:,6575:end);
ssv = ncread('..\ssv_bsc.nc','ssv');
v = ssv(:,:,6575:end);
lon1=173:0.125:192.875;
lat1=52:0.125:61.875;
[y2,x2]=meshgrid(lat1,lon1);
start_date = datenum(2011,01,01);
end_date = datenum(2020,12,31); 
T = start_date:end_date;
load('G:\colorbar\MPL_RdBu_r.mat');
cc = MPL_RdBu_r([1:3:64 end-64:end],:);
zz  =2079;
%% Drawing comparing the CMM and VG algorithms and comparing the number of eddies per day
dd = 4; % Velocity vector arrow interval
d = 0.4; % Velocity vector arrow size
figure
set(gcf,'position',[100   100   1920  1200]);
Sub1 = subplot(2,2,1);

m_proj('equidistant','lon',[173 192.875],'lat',[52 61.875]);
h1 = m_pcolor(x2,y2,squeeze(sla_huatu(zz,:,:)));
set(h1, 'LineStyle','none');
hold on
lons = []; lats = [];
for j = 1:length(shape_new(zz).lonlat)
    lons = shape_new(zz).lonlat{1,j}(1,:);
    lons = smooth(lons,3);
    clon = shape_new(zz).lon(j);
    clat = shape_new(zz).lat(j);
    lats = shape_new(zz).lonlat{1,j}(2,:);
    lats = smooth(lats,3);
    if shape_new(zz).type(j) == 1
        
        m_plot(lons,lats,'Color',[0.019608000,0.18823500,0.38039199],'Marker','.','Linewidth',1.5,'MarkerSize',4);
        hold on
        m_plot(clon,clat,'Color',[0.019608000,0.18823500,0.38039199],'Marker','.','MarkerSize',10)
        hold on
    else
        m_plot(lons,lats,'Color',[102,36,0]/255,'Marker','.','Linewidth',1.5,'MarkerSize',4);
        hold on
        m_plot(clon,clat,'Color',[102,36,0]/255,'Marker','.','MarkerSize',10)
        hold on
    end
end
colormap(cc)
caxis([-0.1 0.3]);
m_gshhs_h('patch',[69 85 89]/255,'EdgeColor',[69 85 89]/255);
set(gca, 'XTick', 150:25:209.875);
m_grid('linest','none','linewid',1.5,'FontSize',10,'FontName','Times New Roman','FontWeight','bold'...
    ,'xtick',([175 179 183 187 191]),'ytick',([52 54 56 58 60]),'fontsize',20);
m_quiver(x2(1:dd:end,1:dd:end),y2(1:dd:end,1:dd:end),u(1:dd:end,1:dd:end,zz)./d,v(1:dd:end,1:dd:end,zz)./d,0,'k');%后面的0决定了箭头是否随图的比例放大缩小
.../10是调整箭头大小
    title('Combined Multi-model','fontsize',20,'FontWeight','bold');
Position_Sub1 = get(Sub1, 'Position');  % Get the position of the submap [x,y,width,height]
                                       % The four values are the proportion of the x and y coordinates, width and height of the lower left corner of this subplot expressed as a percentage
Position_Sub1 = Position_Sub1 + [-0.06 -0.02 0.05 0.05];     % Set the first sub diagram to move to the left                                
set(Sub1, 'Position',Position_Sub1)    % Resetting the position of the first sub-image

Sub2 = subplot(2,2,2);
m_proj('equidistant','lon',[173 192.875],'lat',[52 61.875]);
hold on
h2 = m_pcolor(x2,y2,squeeze(sla_huatu(zz,:,:)));
set(h2, 'LineStyle','none');
hold on
lons = []; lats = [];
for j = 1:length(shape_vg(zz).lonlat)
    lons = shape_vg(zz).lonlat{1,j}(1,:);
    lons = smooth(lons,3);
    clon = shape_vg(zz).lon(j);
    clat = shape_vg(zz).lat(j);
    lats = shape_vg(zz).lonlat{1,j}(2,:);
    lats = smooth(lats,3);
    if shape_vg(zz).type(j) == 1
        
        m_plot(lons,lats,'Color',[0.019608000,0.18823500,0.38039199],'Marker','.','Linewidth',1.5,'MarkerSize',4);
        hold on
        m_plot(clon,clat,'Color',[0.019608000,0.18823500,0.38039199],'Marker','.','MarkerSize',10)
        hold on
    else
        m_plot(lons,lats,'Color',[102,36,0]/255,'Marker','.','Linewidth',1.5,'MarkerSize',4);
        hold on
        m_plot(clon,clat,'Color',[102,36,0]/255,'Marker','.','MarkerSize',10)
        hold on
    end
end

colormap(cc)
caxis([-0.1 0.3]);
m_gshhs_h('patch',[69 85 89]/255,'EdgeColor',[69 85 89]/255);
set(gca, 'XTick', 150:25:209.875);
m_grid('linest','none','linewid',1.5,'FontSize',10,'FontName','Times New Roman','FontWeight','bold'...
    ,'xtick',([175 179 183 187 191]),'ytick',([52 54 56 58 60]),'fontsize',20);
m_quiver(x2(1:dd:end,1:dd:end),y2(1:dd:end,1:dd:end),u(1:dd:end,1:dd:end,zz)./d,v(1:dd:end,1:dd:end,zz)./d,0,'k');
.../10是调整箭头大小
title('VG','fontsize',20,'FontWeight','bold');
Position_Sub2 = get(Sub2, 'Position'); 
                                       
Position_Sub2 = Position_Sub2 + [-0.075 -0.02 0.05 0.05];                                  
set(Sub2, 'Position',Position_Sub2)   

colorbar;
hBar = colorbar;
set(hBar,'ytick',-0.1:0.1:0.3,'FontSize',20);
caxis([-0.1 0.3])
ylabel(hBar,'SLA(m)','FontSize',20,'FontWeight','normal');

Sub3 = subplot(2,2,[3,4]);
plot(T,smooth(new_num,31),'Color',[20 20 15]/255,'Linewidth',1.8); % 31-day moving average
hold on
plot(T,smooth(vg_num,31),'Color',[0 92 83]/255,'Linewidth',1.5);
hold on
plot(T,smooth(att_suv_num,31),'Color',[22 92 171 ]/255,'Linewidth',1.5);
hold on
plot(T,smooth(dan_suv_num,31),'Color',[173 57 152]/255,'Linewidth',1.5);
hold on
plot(T,smooth(en_suv_num,31),'Color',[239 96 36]/255,'Linewidth',1.5);

datetick('x',10);
set(gca,'Ytick',10:5:30,'FontSize',20);
xlabel('Time','FontSize',20,'FontWeight','bold');
ylabel('Number','FontSize',20,'FontWeight','bold');
title('Number of eddies detected','fontsize',20,'FontWeight','bold');
legend('CMM','VG','Att-suv','Dan-suv','En-suv','Location','northeastoutside');
%保存为pdf
set(gcf,'Units','points');
pos = get(gcf,'Position');
set(gcf,'PaperPositionMode','Auto','PaperUnits','points','PaperSize',[pos(3), pos(4)])
filename =['2combine_new',num2str(zz),'.pdf']; % 设定导出文件名
% print(gcf,filename,'-painters','-dpdf')
exportgraphics(gcf,filename,'ContentType','vector')
close(gcf)

%% Save new results
day = 1:3653;
for i = 1:length(day)
    centers(i).day = day(i);
    centersold(i).lat= shape_new(i).lat;
    [~,order] = sort(centersold(i).lat,2);
    centersold(i).type = shape_new(i).type;
    centers(i).type = centersold(i).type(order);
    centers(i).lat = centersold(i).lat(order);
    centersold(i).lon = shape_new(i).lon;
    centers(i).lon =  centersold(i).lon(order);
    centersold(i).j = shape_new(i).j;
    centers(i).j = centersold(i).j(order);
    centersold(i).i = shape_new(i).i;
    centers(i).i = centersold(i).i(order);
    shapesold(i).lonlat = shape_new(i).lonlat;
    shapes(i).lonlat = shapesold(i).lonlat(order);
    
end
save('E:\term\eddy_matlab\wenzhanghuatu\track_data_new\eddy_centers.mat','centers');
save('E:\term\eddy_matlab\wenzhanghuatu\track_data_new\eddy_shapes.mat','shapes');