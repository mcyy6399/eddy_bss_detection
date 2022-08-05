clc;clear ;

%This script is used to interpolate SLA
% lon_lat_grid
lon_v = double(ncread('G:\alldata\AVISO_2021\2009\02\dt_global_allsat_phy_l4_20090202_20210726.nc','longitude'));
lat_v = double(ncread('G:\alldata\AVISO_2021\2009\02\dt_global_allsat_phy_l4_20090202_20210726.nc','latitude'));
lon_v = [lon_v(721:end);lon_v(1:720)+360];
lon1=173:0.125:192.875;
lat1=52:0.125:61.875;
indexlon=692:772;
indexlat=568:608;
long=lon_v(indexlon);
latt=lat_v(indexlat);
[y,x]=meshgrid(latt,long);
[y2,x2]=meshgrid(lat1,lon1);
avisodir=('G:\alldata\AVISO_2021\');
avisolist=dir(avisodir);
sla_grid = [];
for i = 3:length(avisolist)

subdir=fullfile(avisodir,avisolist(i).name);
avisodata = dir(subdir);
ndata = length(avisodata);
for j = 3:ndata
monthname = fullfile(subdir,avisodata(j).name);
monthdir = dir(monthname);
nsla = length(monthdir);
sla_new = [];
parfor k=3:nsla
filename = fullfile(monthname,monthdir(k).name);
sla = ncread(filename,'sla');
sla2 = [sla(721:end,:);sla(1:720,:)];
sla_old = sla2(indexlon,indexlat);
sla_new(k-2,:,:) = griddata(x,y,sla_old,x2,y2);% 1/4 to 1/8
end
sla_grid = cat(1,sla_grid,sla_new);
end
end
%% save sla as nc
file='E:\term\mesoscale_eddy\sla_concat\eddy_sla.nc';
if exist(file,'file')
    eval(['delete ', file ])
end
start_date = datenum(1993,01,01);
end_date = datenum(2020,12,31);
T = start_date:end_date;
[t,m,n]=size(sla_grid);

nccreate(file,'day','dimensions',{'day' t },'format','classic');
nccreate(file,'longitude','dimensions',{ 'lon' m 'lat' n },'format','classic') ; 
nccreate(file,'latitude','dimensions',{'lon' m 'lat' n },'format','classic') ; 
nccreate(file,'sla','dimensions',{ 'day' t 'lon' m 'lat' n },'format','classic') ; 
ncwrite(file,'sla',sla_grid);
ncwrite(file,'day',T);
ncwrite(file,'longitude',x2);
ncwrite(file,'latitude',y2);
clear m n t