clc;clear

% This script is to prepare the input data for the VG algorithm.
% Data from AVISO's Delayed-Time Level-4 sea surface height
cd 'E:\term\eddy_matlab\'
lon = double(ncread('G:\alldata\AVISO_2021\2009\02\dt_global_allsat_phy_l4_20090202_20210726.nc','longitude'));% load lonlat information
lat = double(ncread('G:\alldata\AVISO_2021\2009\02\dt_global_allsat_phy_l4_20090202_20210726.nc','latitude'));
lon = [lon(721:end);lon(1:720)+360];% -180:180 to 0-360
lon1=173:0.125:192.875; % bering sea slope
lat1=52:0.125:61.875;
indexlon=692:772;
indexlat=568:608;
long=lon(indexlon);
latt=lat(indexlat);
[y,x]=meshgrid(latt,long);
[y2,x2]=meshgrid(lat1,lon1);

avisodir=('G:\alldata\AVISO_2021\');
avisolist=dir(avisodir);%Traversing the entire folder

ssu = [];ssv = [];
avisodata = [];
for i = 3:length(avisolist)

subdir=fullfile(avisodir,avisolist(i).name);
avisodata = dir(subdir);
ndata = length(avisodata);

for j=3:ndata
monthname = fullfile(avisodir,avisolist(i).name,avisodata(j).name);
monthdir = dir(monthname);
nuv = length(monthdir);
ugos_new = [];
vgos_new = [];
parfor k = 3:nuv
filename = fullfile(monthname,monthdir(k).name)
ugos = ncread(filename,'ugos');
ugos = [ugos(721:end,:);ugos(1:720,:)];
vgos = ncread(filename,'vgos');
vgos = [vgos(721:end,:);vgos(1:720,:)];
ugos_old = ugos(indexlon,indexlat);
vgos_old = vgos(indexlon,indexlat);
ugos_new(:,:,k-2) = griddata(x,y,ugos_old,x2,y2);
vgos_new(:,:,k-2) = griddata(x,y,vgos_old,x2,y2);
end
ssu = cat(3,ssu,ugos_new);
ssv = cat(3,ssv,vgos_new);
end

end
speed=sqrt(ssu.^2+ssv.^2);
for i=1:160
    for j=1:80
            if isnan(speed(i,j,:))
                    mask(i,j)=0;
            else
                    mask(i,j)=1;
            end
    end
end
%%
T = 1:10227';
file='ssu_bsc.nc';
if exist(file,'file')
    eval(['delete ', file ])
end

[m,n,t]=size(ssu);

nccreate(file,'day','dimensions',{'day' t},'format','classic');
nccreate(file,'ssu','dimensions',{ 'lon' m  'lat' n  'day' t },'format','classic') ; 
ncwrite(file,'ssu',ssu);
ncwrite(file,'day',T);
clear m n t
%%
file='ssv_bsc.nc';
if exist(file,'file')
    eval(['delete ', file ])
end
[m,n,t]=size(ssv);

nccreate(file,'day','dimensions',{'day' t},'format','classic');
nccreate(file,'ssv','dimensions',{ 'lon' m  'lat' n  'day' t },'format','classic') ; 
ncwrite(file,'ssv',ssv);
ncwrite(file,'day',T);
clear m n t

%%
file='lon_lat_bsc.nc';
if exist(file,'file')
    eval(['delete ', file ])
end

[m,n]=size(mask);
nccreate(file,'lon','dimensions',{'lon' m  'lat' n },'format','classic') ; 
nccreate(file,'lat','dimensions',{'lon' m  'lat' n },'format','classic') ; 
nccreate(file,'mask','dimensions',{'lon' m  'lat' n },'format','classic') ; 
ncwrite(file,'lon',x2);
ncwrite(file,'lat',y2);
ncwrite(file,'mask',mask);  
