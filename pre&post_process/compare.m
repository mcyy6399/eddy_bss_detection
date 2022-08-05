clc;clear

% Comparison of the oceanic eddies detected by the four different algorithms in the BSs region on 09 September 2016

% load sla data
cd 'E:\term\eddy_matlab\'
sla = ncread('E:\term\mesoscale_eddy\sla_concat\eddy_sla.nc','sla');
sla_huatu = sla(6575:end,:,:);
ssu = ncread('..\ssu_bsc.nc','ssu');
u = ssu(:,:,6575:end);
ssv = ncread('..\ssv_bsc.nc','ssv');
v = ssv(:,:,6575:end);

% Loading the data from modeltrack results
shape_dan_sla = loadcs('..\track_data_dan_onlysla2\',1);
shape_att_sla = loadcs('..\track_data_att_onlysla3\',1);
shape_en_sla = loadcs('..\track_data_en_onlysla3\',1);
shape_dan_suv = loadcs('..\track_data_dan_suv2\',1);
shape_att_suv = loadcs('..\track_data_att_suv3\',1);
shape_en_suv = loadcs('..\track_data_en_suv3\',1);
shape_vg = loadcs('..\vg_track_data2\',1);

lon1=173:0.125:192.875;
lat1=52:0.125:61.875;
[y2,x2]=meshgrid(lat1,lon1);
start_date = datenum(2011,01,01);
end_date = datenum(2020,12,31);
T = start_date:end_date;

load('G:\colorbar\MPL_RdBu_r.mat'); % color setup
cc = MPL_RdBu_r([1:3:64 end-64:end],:);

%%
% parfor zz = 2000:2500
zz = 2079;
dd1 = [0  0.0   0.045  0.045 ];
dd2 = [0  -0.01   0.045  0.045 ];
dd3 = [0  -0.02   0.045  0.045 ];
dd4 = [0  -0.03  0.045  0.045 ];
figure('visible','off');
% figure
set(gcf,'position',[100   100   1200  3200]);
Sub1 = subplot(4,2,1);  
picture(shape_vg,sla_huatu,x2,y2,u,v,zz,cc,'VG',1)
Position_Sub1 = get(Sub1, 'Position');  % Get the location of the subplot[x,y,width,height].
                                        % The four values are the proportion
                                        % of the x and y coordinates, width and height of the lower left corner of this subplot expressed as a percentage.
Position_Sub1 = Position_Sub1 + dd1;    % Set the sub diagram to move to the left.                                
set(Sub1, 'Position',Position_Sub1)     % Reset the position of the sub diagram.

Sub2 = subplot(4,2,3);
picture(shape_att_sla,sla_huatu,x2,y2,u,v,zz,cc,'Att(SLA)',1)
Position_Sub2 = get(Sub2, 'Position');
                                      
Position_Sub2 = Position_Sub2 + dd2;                               
set(Sub2, 'Position',Position_Sub2)     

Sub3 = subplot(4,2,5);
picture(shape_dan_sla,sla_huatu,x2,y2,u,v,zz,cc,'Dan(SLA)',1)
Position_Sub3 = get(Sub3, 'Position');
                                      
Position_Sub3 = Position_Sub3 + dd3;                         
set(Sub3, 'Position',Position_Sub3)

Sub4 = subplot(4,2,7);
picture(shape_en_sla,sla_huatu,x2,y2,u,v,zz,cc,'En(SLA)',2)
Position_Sub4 = get(Sub4, 'Position');
                                     
Position_Sub4 = Position_Sub4 + dd4;                         
set(Sub4, 'Position',Position_Sub4)

Sub5 = subplot(4,2,2);
picture(shape_vg,sla_huatu,x2,y2,u,v,zz,cc,'VG',3)
Position_Sub5 = get(Sub5, 'Position');
                                      
Position_Sub5 = Position_Sub5 + dd1;                             
Position_Sub5(1)=Position_Sub5(1)-0.04;
set(Sub5, 'Position',Position_Sub5)

Sub6 = subplot(4,2,4);
picture(shape_att_suv,sla_huatu,x2,y2,u,v,zz,cc,'Att(SLA,U,V)',3)
Position_Sub6 = get(Sub6, 'Position');  
                                     
Position_Sub6 = Position_Sub6 + dd2;                              
Position_Sub6(1)=Position_Sub6(1)-0.04;
set(Sub6, 'Position',Position_Sub6)  

Sub7 = subplot(4,2,6);
picture(shape_dan_suv,sla_huatu,x2,y2,u,v,zz,cc,'Dan(SLA,U,V)',3)
Position_Sub7 = get(Sub7, 'Position'); 
                                      
Position_Sub7 = Position_Sub7 + dd3;                         
Position_Sub7(1)=Position_Sub7(1)-0.04;
set(Sub7, 'Position',Position_Sub7)   

Sub8 = subplot(4,2,8);
picture(shape_en_suv,sla_huatu,x2,y2,u,v,zz,cc,'En(SLA,U,V)',4)
Position_Sub8 = get(Sub8, 'Position'); 
                                      
Position_Sub8 = Position_Sub8 + dd4;                              
Position_Sub8(1)=Position_Sub8(1)-0.04;
set(Sub8, 'Position',Position_Sub8)   
% 
hBar = colorbar('southoutside');
caxis([-0.1 0.3]);
set(hBar,'ytick',-0.1:0.1:0.3,'FontSize',20);

ylabel(hBar,'SLA(m)','FontSize',20,'FontWeight','normal');
Position_Bar = get(hBar, 'Position');    % Get the position of the colorbar [x,y,width,height]
Position_Bar = Position_Bar + [-0.4 -0.065  0.4  0.005 ];     % Set colorbar to move to the right and widen                                  
set(hBar, 'Position',Position_Bar) % Resetting the position of the colorbar  
get(hBar, 'Position');

% save figure as pdf(vector)
set(gcf,'Units','points');
pos = get(gcf,'Position');
set(gcf,'PaperPositionMode','Auto','PaperUnits','points','PaperSize',[pos(3), pos(4)])
filename = ['E:\term\eddy_matlab\wenzhanghuatu\',['acompare_new',num2str(zz),'.pdf']]; 
% print(gcf,filename,'-painters','-dpdf')
exportgraphics(gcf,filename,'ContentType','vector')
% exportgraphics(gcf,filename);
close(gcf)
% end
