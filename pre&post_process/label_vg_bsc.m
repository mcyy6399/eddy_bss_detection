clc; clear 

% This script was used to make the VG algorithm's eddy detection result into a label for model training.
% Anticyclonic eddies are marked as 1, cyclonic eddies as 2, and land and non-eddy areas as 0.
sla = ncread('E:\term\mesoscale_eddy\sla_concat\eddy_sla.nc','sla');
x = ncread('E:\term\mesoscale_eddy\sla_concat\eddy_sla.nc','longitude');
y = ncread('E:\term\mesoscale_eddy\sla_concat\eddy_sla.nc','latitude');
load('E:\term\eddy_matlab\vg_track_data\eddy_centers.mat');
load('E:\term\eddy_matlab\vg_track_data\eddy_shapes.mat');
parfor ii = 1:length(shapes)
shape(ii).type = centers(ii).type;
shape(ii).lonlat = shapes(ii).lonlat;
end
label = zeros(length(shape),160,80);
for i = 1:length(shape)
    sp=[];
    sp=shape(i);
    for j = 1:length(sp.lonlat)
        sp1 = [];sp2 = [];lon = [];lat = [];yy = [];xx = [];
        sp1=sp.lonlat(j);
        sp2=cell2mat(sp1);
        lon=(sp2(1,:))';
        lat=(sp2(2,:))';
        for k =1:length(lon)
            [~,loc_x] = min(abs(min(x-lon(k),[],2)));
            [~,loc_y] = min(abs(min(y-lat(k),[],1)));
            if(sp.type(j)==1)
                label(i,loc_x,loc_y) = 1;
            elseif(sp.type(j)==-1)
                label(i,loc_x,loc_y) = 2;
            end
        end

        if(sp.type(j)==1)
            label(i,inpolygon(x,y,lon,lat)) = 1;
        elseif(sp.type(j)==-1)
            label(i,inpolygon(x,y,lon,lat)) = 2;
        end
    end
end
file='E:\term\mesoscale_eddy\sla_concat\eddy_label.nc';
if exist(file,'file')
    eval(['delete ', file ])
end
start_date = datenum(1993,01,01);
end_date = datenum(2020,12,31); 
T = start_date:end_date;
[t,m,n]=size(label);

nccreate(file,'day','dimensions',{'day' t },'format','classic');
nccreate(file,'longitude','dimensions',{ 'lon' m 'lat' n },'format','classic') ; 
nccreate(file,'latitude','dimensions',{'lon' m 'lat' n },'format','classic') ; 
nccreate(file,'label','dimensions',{'day' t 'lon' m 'lat' n },'format','classic') ; 
ncwrite(file,'label',label);
ncwrite(file,'day',T);
ncwrite(file,'longitude',x);
ncwrite(file,'latitude',y);


% 
% for zz = 1:365
%     figure('visible','off');
%     m_proj('equidistant','lon',[173 193],'lat',[52 62]);
%     
%     hold on
%     h = m_pcolor(x,y,squeeze(sla_grid(zz,:,:)));
%     set(h, 'LineStyle','none');
%         hc=colorbar;
%     set(hc,'ytick',-0.3:0.1:0.3,'FontSize',10);
%     caxis([-0.3 0.3])
%     hold on
%     m_contour(x,y,squeeze(label(zz,:,:)));
%     m_gshhs_h('patch',[236 201 142]/255);
%     m_grid('box','on','tickdir','out');
%     title_str1=strcat('第',num2str(zz),'天');
%     print(gcf,'-dpng',['E:\term\eddy_matlab\labelfig\',['第',num2str(zz),'天labelssh图.png']]);
% end