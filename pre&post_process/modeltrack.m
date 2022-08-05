clc;clear

% This script is used to convert the classification results from the model output into eddy coverage areas.
% This script also saves the results in a manner that the Lagrangian method can understand.

load('E:\term\eddy_matlab\wenzhanghuatu\eddyatt_onlysla.mat');
LON=173:0.125:192.875;
LAT=52:0.125:61.875;
numanti_pre = [];numcyc_pre = [];Eanti_pre = {};Ecyc_pre = {};
for i = 1:3653 % Using the bwboundaries function to convert classification results to binary images and determine the boundaries of eddies.
    anti = double(squeeze(eddy(i,:,:)));
    cyc = double(squeeze(eddy(i,:,:)));
    anti(anti==1) = 0;
    cyc(cyc==2) = 0;
    eddyanti = mat2gray(anti);
    eddyantiim = im2bw(eddyanti,0.4);
    eddycyc = mat2gray(cyc);
    eddycycim = im2bw(cyc,0.4);
    [Banti_pre,Lanti,nanti,Aanti] = bwboundaries(eddyantiim,8,'noholes');
    [Bcyc_pre,Lcyc,ncyc,Acyc] = bwboundaries(eddycycim,8,'noholes');
    n = 0;
    for k = 1:length(Banti_pre) % Determining the latitude, longitude, and centre of the anticyclone boundary.
        if length(Banti_pre{k,1})>2
        n = n+1;
        Eanti_pre(i).lonlat{1,n}(1,:) = LON(Banti_pre{k,1}(:,2));
        Eanti_pre(i).lonlat{1,n}(2,:) = LAT(Banti_pre{k,1}(:,1));
        Eanti_pre(i).type{1,n}(:,1) = -1;
        Eanti_pre(i).i{1,n}= round(sum(Banti_pre{k,1}(1:end-1,2))/(length(Banti_pre{k,1})-1));
        Eanti_pre(i).j{1,n}= round(sum(Banti_pre{k,1}(1:end-1,1))/(length(Banti_pre{k,1})-1));
        Eanti_pre(i).lon{1,n}(:,1) = LON(round(sum(Banti_pre{k,1}(1:end-1,2))/(length(Banti_pre{k,1})-1)));
        Eanti_pre(i).lat{1,n}(:,1) = LAT(round(sum(Banti_pre{k,1}(1:end-1,1))/(length(Banti_pre{k,1})-1)));
        end
    end
    n = 0;
    for k = 1:length(Bcyc_pre) % Determining the latitude, longitude, and centre of the cyclone's boundary.
        if length(Bcyc_pre{k,1})>2
        n = n+1;
        Ecyc_pre(i).lonlat{1,n}(1,:) = LON(Bcyc_pre{k,1}(:,2));
        Ecyc_pre(i).lonlat{1,n}(2,:) = LAT(Bcyc_pre{k,1}(:,1));
        Ecyc_pre(i).type{1,n}(:,1) = 1;
        Ecyc_pre(i).i{1,n}= round(sum(Bcyc_pre{k,1}(1:end-1,2))/(length(Bcyc_pre{k,1})-1));
        Ecyc_pre(i).j{1,n}= round(sum(Bcyc_pre{k,1}(1:end-1,1))/(length(Bcyc_pre{k,1})-1));
        Ecyc_pre(i).lon{1,n}(:,1) = LON(round(sum(Bcyc_pre{k,1}(1:end-1,2))/(length(Bcyc_pre{k,1})-1)));
        Ecyc_pre(i).lat{1,n}(:,1) = LAT(round(sum(Bcyc_pre{k,1}(1:end-1,1))/(length(Bcyc_pre{k,1})-1)));
        end
    end
end

day = 1:3653;
for i = 1:length(day)
    centers(i).day = day(i);
    centersold(i).lat= cat(2,cell2mat(Eanti_pre(i).lat),cell2mat(Ecyc_pre(i).lat));
    [~,order] = sort(centersold(i).lat,2);
    centersold(i).type = cat(2,cell2mat(Eanti_pre(i).type),cell2mat(Ecyc_pre(i).type));
    centers(i).type = centersold(i).type(order);
    centers(i).lat = centersold(i).lat(order);
    centersold(i).lon = cat(2,cell2mat(Eanti_pre(i).lon),cell2mat(Ecyc_pre(i).lon));
    centers(i).lon =  centersold(i).lon(order);
    centersold(i).j = cat(2,cell2mat(Eanti_pre(i).j),cell2mat(Ecyc_pre(i).j));
    centers(i).j = centersold(i).j(order);
    centersold(i).i = cat(2,cell2mat(Eanti_pre(i).i),cell2mat(Ecyc_pre(i).i));
    centers(i).i = centersold(i).i(order);
    shapesold(i).lonlat = [Eanti_pre(i).lonlat Ecyc_pre(i).lonlat];
    shapes(i).lonlat = shapesold(i).lonlat(order);
    
end
save('E:\term\eddy_matlab\track_data_en_onlysla3\eddy_centers.mat','centers');
save('E:\term\eddy_matlab\track_data_en_onlysla3\eddy_shapes.mat','shapes');
%%
cd 'E:\eddymatlab\'
lon_v = double(ncread('G:\alldata\AVISO\2000\dt_global_allsat_phy_l4_20000101_20170110.nc','longitude'));
lat_v = double(ncread('G:\alldata\AVISO\2000\dt_global_allsat_phy_l4_20000101_20170110.nc','latitude'));
lon1=150:0.125:209.875;
lat1=45:0.125:64.875;
indexlon=600:840; 
indexlat=540:620; 
lon_old=lon_v(indexlon);
lat_old=lat_v(indexlat);
[y,x]=meshgrid(lat_old,lon_old);
[y2,x2]=meshgrid(lat1,lon1);
path = 'G:\alldata\AVISO\2018\';
maindir = 'G:\alldata\AVISO\2018\*.nc';
dirs=dir(maindir);          %读取该路径所有nc文件
dirnum=length(dirs); %计算文件夹里文档的个数
sla_new = [];
for i=1:1
filename = fullfile(path,dirs(i).name);
sla = ncread(filename,'sla');
sla_old = sla(indexlon,indexlat);
sla_new(:,:,i) = griddata(x,y,sla_old,x2,y2);
end
for i=1:480
    for j=1:160
            if isnan(sla_new(i,j,:))
                    mask(i,j)=0;
            else
                    mask(i,j)=1;
            end
    end
end


file='lon_lat_480.nc';
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
%%
cd 'E:\term\eddy_matlab\'
nc_dim='lon_lat_bsc.nc';
r = 22;
path_out = 'E:\term\eddy_matlab\vg_track_data2\' ;

mod_eddy_tracks(nc_dim,r,path_out)
