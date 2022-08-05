clc;clear;

% This script is used to count the radius and lifetime of the eddy. (A circle with a radius equal to the equal area of the eddy)

% Calculate the actual area of each grid point
lon1=173:0.125:192.875;
lat1=52:0.125:61.875;
[y2,x2]=meshgrid(lat1,lon1);
[~,dx] = gradient(x2);
dx = dx.*(pi./180).*R.*cos(y2*pi./180);
[dy,~] = gradient(y2);
dy = dy.*(pi./180).*R;
area = dx.*dy;
% Loading eddy shape and eddy centre data
load('.\track_data_new\eddy_shapes.mat');
load('.\track_data_new\eddy_centers.mat');
shapes_new = shapes;
centers_new = centers;
clear shapes centers
load('..\vg_track_data\eddy_shapes.mat');
load('..\vg_track_data\eddy_centers.mat');
shapes_vg = shapes(6575:end);
centers_vg = centers(6575:end);
clear shapes

r_anti = zeros(1,5); r_cyc = zeros(1,5);
R_anti = [];R_cyc = [];
for i = 1:length(shapes_new)
    
    lonlat = shapes_new(i).lonlat;
    
    for j = 1:length(lonlat)
        ty = centers_new(i).type(j);
        lons  = lonlat{1,j}(1,:);
        lats = lonlat{1,j}(2,:);
        [in,on]= inpolygon(x2,y2,lons,lats); % Grid points within the eddy and at the boundaries.
        r = sqrt((sum(sum(area(in)))+sum(sum(area(on))))/pi)/1000;% Calculating the eddy area
        if ty == -1 % Statistical anticyclone radius distribution
            R_anti = [R_anti r];
            if r<=25
                r_anti(1) = r_anti(1) + 1;
            elseif r>25&&r<50
                r_anti(2) = r_anti(2) + 1;
            elseif r>50&&r<75
                r_anti(3) = r_anti(3) + 1;
            elseif r>75&&r<100
                r_anti(4) = r_anti(4) + 1;
            elseif r>100
                r_anti(5) = r_anti(5) + 1;
 
            end
        else % Statistical cyclone radius distribution
            R_cyc = [R_cyc r];
            if r<=25
                r_cyc(1) = r_cyc(1) + 1;
            elseif r>25&&r<50
                r_cyc(2) = r_cyc(2) + 1;
            elseif r>50&&r<75
                r_cyc(3) = r_cyc(3) + 1;
            elseif r>75&&r<100
                r_cyc(4) = r_cyc(4) + 1;
            elseif r>100
                r_cyc(5) = r_cyc(5) + 1;
            end
        end
        
    end
end


r_anti_vg = zeros(1,5); r_cyc_vg = zeros(1,5);

R_anti_vg = [];R_cyc_vg = [];
for i = 1:length(shapes_vg)
    
    lonlat = shapes_vg(i).lonlat;
    
    for j = 1:length(lonlat)
        ty = centers_vg(i).type(j);
        lons  = lonlat{1,j}(1,:);
        lats = lonlat{1,j}(2,:);
        [in,on]= inpolygon(x2,y2,lons,lats);
        r = sqrt((sum(sum(area(in)))+sum(sum(area(on))))/pi)/1000;
        if ty == -1
            R_anti_vg = [R_anti_vg r];
            if r<=25
                r_anti_vg(1) = r_anti_vg(1) + 1;
            elseif r>25&&r<50
                r_anti_vg(2) = r_anti_vg(2) + 1;
            elseif r>50&&r<75
                r_anti_vg(3) = r_anti_vg(3) + 1;
            elseif r>75&&r<100
                r_anti_vg(4) = r_anti_vg(4) + 1;
            elseif r>100
                r_anti_vg(5) = r_anti_vg(5) + 1;
            end
        else
            R_cyc_vg = [R_cyc_vg r];
            if r<=25
                r_cyc_vg(1) = r_cyc_vg(1) + 1;
            elseif r>25&&r<50
                r_cyc_vg(2) = r_cyc_vg(2) + 1;
            elseif r>50&&r<75
                r_cyc_vg(3) = r_cyc_vg(3) + 1;
            elseif r>75&&r<100
                r_cyc_vg(4) = r_cyc_vg(4) + 1;
            elseif r>100
                r_cyc_vg(5) = r_cyc_vg(5) + 1;
            end
        end
    end
    
end
% Create horizontal coordinates and radius arrays
dd = zeros(1,5);
r_anti_all = [flip(r_anti_vg);flip(r_anti)];
r_cyc_all = [r_cyc;r_cyc_vg];
r_all = [r_anti_all [0;0] r_cyc_all];
x1 = (-5:-1);
x2 = (1:5);
%% Plotting radius bars
figure
set(gcf,'position',[100   100   1920  1200]);
b1 = bar(x1,r_anti_all,0.95);
b1(1).FaceColor = 'flat';
b1(1).CData = [253 217 159]/255;
b1(2).FaceColor = 'flat';
b1(2).CData = [242 132 66]/255;
hold on
b2 = bar(x2,r_cyc_all,0.95);
b2(1).FaceColor = 'flat';
b2(1).CData = [0 131 141]/255;
b2(2).FaceColor = 'flat';
b2(2).CData = [192 209 184]/255;
xlim = ([-15 15]);
set(gca,'Xtick',-5:1:5,'Xticklabel',{'>100km' , '75-100km' ,'50-75km' ,'25-50km', '0-25km','' ,'0-25km' ,'25-50km' ,'50-75km', '75-100km','>100km'},'FontSize',20);

    legend('Anti-VG','Anti-CMM','Cyc-CMM','Cyc-VG','Fontsize',20)
    set(gca,'Ytick',0:5e3:2.5e4);
xlabel('Radius','FontSize',20,'FontWeight','bold');
ylabel('Number','FontSize',20,'FontWeight','bold');
title('Radius distribution of the eddyies','fontsize',20,'FontWeight','bold');

set(gcf,'Units','points');
pos = get(gcf,'Position');
set(gcf,'PaperPositionMode','Auto','PaperUnits','points','PaperSize',[pos(3), pos(4)])
filename = 'radius_new2.pdf'; % Set the export file name
% print(gcf,filename,'-painters','-dpdf')
exportgraphics(gcf,filename,'ContentType','vector')
close(gcf)
%% lifetime
% for eddies with lifetime>4 weeks
load('.\track_data_new\eddy_tracks.mat');
anti_lt = zeros(1,15);cyc_lt = zeros(1,15);
anti_lt_all = [];cyc_lt_all = [];
for i = 1:length(tracks)
    days = length(tracks(i).day);
    ty = tracks(i).type(1);
    if ty == -1
        anti_lt_all = [anti_lt_all days];
        if days>28&&days<=35
            anti_lt(1) = anti_lt(1)+1;
        elseif days>35&&days<=42
            anti_lt(2) = anti_lt(2)+1;
        elseif days>42&&days<=49
            anti_lt(3) = anti_lt(3)+1;
        elseif days>49&&days<=56
            anti_lt(4) = anti_lt(4)+1;
        elseif days>56&&days<=63
            anti_lt(5) = anti_lt(5)+1;
        elseif days>63&&days<=70
            anti_lt(6) = anti_lt(6)+1;
        elseif days>77&&days<=84
            anti_lt(7) = anti_lt(7)+1;
        elseif days>84&&days<=91
            anti_lt(8) = anti_lt(8)+1;
        elseif days>91&&days<=98
            anti_lt(9) = anti_lt(9)+1;
        elseif days>105&&days<=112
            anti_lt(10) = anti_lt(10)+1;
        elseif days>112&&days<=119
            anti_lt(11) = anti_lt(11)+1;
        elseif days>119&&days<=126
            anti_lt(12) = anti_lt(12)+1;
        elseif days>126&&days<=133
            anti_lt(13) = anti_lt(13)+1;
        elseif days>133&&days<=140
            anti_lt(14) = anti_lt(14)+1;
        elseif days>140
            anti_lt(15) = anti_lt(15)+1;
        end
    else
        cyc_lt_all = [cyc_lt_all days];
        if days>28&&days<=35
            cyc_lt(1) = cyc_lt(1)+1;
        elseif days>35&&days<=42
            cyc_lt(2) = cyc_lt(2)+1;
        elseif days>42&&days<=49
            cyc_lt(3) = cyc_lt(3)+1;
        elseif days>49&&days<=56
            cyc_lt(4) = cyc_lt(4)+1;
        elseif days>56&&days<=63
            cyc_lt(5) = cyc_lt(5)+1;
        elseif days>63&&days<=70
            cyc_lt(6) = cyc_lt(6)+1;
        elseif days>77&&days<=84
            cyc_lt(7) = cyc_lt(7)+1;
        elseif days>84&&days<=91
            cyc_lt(8) = cyc_lt(8)+1;
        elseif days>91&&days<=98
            cyc_lt(9) = cyc_lt(9)+1;
        elseif days>105&&days<=112
            cyc_lt(10) = cyc_lt(10)+1;
        elseif days>112&&days<=119
            cyc_lt(11) = cyc_lt(11)+1;
        elseif days>119&&days<=126
            cyc_lt(12) = cyc_lt(12)+1;
        elseif days>126&&days<=133
            cyc_lt(13) = cyc_lt(13)+1;
        elseif days>133&&days<=140
            cyc_lt(14) = cyc_lt(14)+1;
        elseif days>140
            cyc_lt(15) = cyc_lt(15)+1;
        end
    end
end
clear tracks
load('..\vg_track_data2\eddy_tracks.mat');

anti_lt_vg = zeros(1,15);cyc_lt_vg = zeros(1,15);
 anti_lt_all_vg = []; cyc_lt_all_vg = [];
for i = 1:length(tracks)
    days = length(tracks(i).day);
    ty = tracks(i).type(1);
    if ty == -1
        anti_lt_all_vg = [anti_lt_all_vg days];
        if days>28&&days<=35
            anti_lt_vg(1) = anti_lt_vg(1)+1;
        elseif days>35&&days<=42
            anti_lt_vg(2) = anti_lt_vg(2)+1;
        elseif days>42&&days<=49
            anti_lt_vg(3) = anti_lt_vg(3)+1;
        elseif days>49&&days<=56
            anti_lt_vg(4) = anti_lt_vg(4)+1;
        elseif days>56&&days<=63
            anti_lt_vg(5) = anti_lt_vg(5)+1;
        elseif days>63&&days<=70
            anti_lt_vg(6) = anti_lt_vg(6)+1;
        elseif days>77&&days<=84
            anti_lt_vg(7) = anti_lt_vg(7)+1;
        elseif days>84&&days<=91
            anti_lt_vg(8) = anti_lt_vg(8)+1;
        elseif days>91&&days<=98
            anti_lt_vg(9) = anti_lt_vg(9)+1;
        elseif days>105&&days<=112
            anti_lt_vg(10) = anti_lt_vg(10)+1;
        elseif days>112&&days<=119
            anti_lt_vg(11) = anti_lt_vg(11)+1;
        elseif days>119&&days<=126
            anti_lt_vg(12) = anti_lt_vg(12)+1;
        elseif days>126&&days<=133
            anti_lt_vg(13) = anti_lt_vg(13)+1;
        elseif days>133&&days<=140
            anti_lt_vg(14) = anti_lt_vg(14)+1;
        elseif days>140
            anti_lt_vg(15) = anti_lt_vg(15)+1;
        end
    else
        cyc_lt_all_vg = [cyc_lt_all_vg days];
        if days>28&&days<=35
            cyc_lt_vg(1) = cyc_lt_vg(1)+1;
        elseif days>35&&days<=42
            cyc_lt_vg(2) = cyc_lt_vg(2)+1;
        elseif days>42&&days<=49
            cyc_lt_vg(3) = cyc_lt_vg(3)+1;
        elseif days>49&&days<=56
            cyc_lt_vg(4) = cyc_lt_vg(4)+1;
        elseif days>56&&days<=63
            cyc_lt_vg(5) = cyc_lt_vg(5)+1;
        elseif days>63&&days<=70
            cyc_lt_vg(6) = cyc_lt_vg(6)+1;
        elseif days>77&&days<=84
            cyc_lt_vg(7) = cyc_lt_vg(7)+1;
        elseif days>84&&days<=91
            cyc_lt_vg(8) = cyc_lt_vg(8)+1;
        elseif days>91&&days<=98
            cyc_lt_vg(9) = cyc_lt_vg(9)+1;
        elseif days>105&&days<=112
            cyc_lt_vg(10) = cyc_lt_vg(10)+1;
        elseif days>112&&days<=119
            cyc_lt_vg(11) = cyc_lt_vg(11)+1;
        elseif days>119&&days<=126
            cyc_lt_vg(12) = cyc_lt_vg(12)+1;
        elseif days>126&&days<=133
            cyc_lt_vg(13) = cyc_lt_vg(13)+1;
        elseif days>133&&days<=140
            cyc_lt_vg(14) = cyc_lt_vg(14)+1;
        elseif days>140
            cyc_lt_vg(15) = cyc_lt_vg(15)+1;
 
        end
    end
end
%%

x1 = (-16:-2)';
x2 = (2:16)';

anti_lt_r = flip(anti_lt);
anti_lt_vg_r = flip(anti_lt_vg);
figure
set(gcf,'position',[100   100   1920   1200]);
Sub1 = subplot(2,1,1);
b1 = bar(x1,anti_lt_r,'Facecolor',[242 132 66]/255);
hold on
b2 = bar(x2,cyc_lt,'Facecolor',[0 131 141]/255);
ylim([0 100]);
xlabel('Lifetime(weeks)','FontSize',205,'FontWeight','bold');
ylabel('Number','FontSize',20,'FontWeight','bold');
title('CMM','fontsize',20,'FontWeight','bold');
set(gca,'Xtick',-16:16,'Xticklabel',{ '>19' ,'18', '17', '16', '15', '14', '13','12','11','10','9','8','7','6','5','','','','5','6','7','8','9','10','11','12','13','14','15','16','17','18','>19'},'FontSize',20);
legend('Anticyclonic','Cyclonic','Fontsize',20)
Position_Sub1 = get(Sub1, 'Position'); 
                                      
Position_Sub1 = Position_Sub1 + [-0.025 -0.02 0.05 -0.05];                                   
set(Sub1, 'Position',Position_Sub1)     

Sub2 = subplot(2,1,2);

b3 = bar(x1,anti_lt_vg_r,'Facecolor',[242 132 66]/255);
hold on
b4 = bar(x2,cyc_lt_vg,'Facecolor',[0 131 141]/255);
ylim([0 100]);

xlabel('Lifetime(weeks)','FontSize',15,'FontWeight','bold');
ylabel('Number','FontSize',20,'FontWeight','bold');
title('vg','fontsize',20,'FontWeight','bold');
set(gca,'Xtick',-16:16,'Xticklabel',{ '>19' ,'18', '17', '16', '15', '14', '13','12','11','10','9','8','7','6','5','','','','5','6','7','8','9','10','11','12','13','14','15','16','17','18','>19'},'FontSize',20);

legend('Anticyclonic','Cyclonic','Fontsize',20)
Position_Sub2 = get(Sub2, 'Position');  
                                       
Position_Sub2 = Position_Sub2 + [-0.025 -0.02 0.05 0];                                  
set(Sub2, 'Position',Position_Sub2)     
sgtitle('Lifetime distribution of the eddyies','fontsize',20,'FontWeight','bold');

set(gcf,'Units','points');
pos = get(gcf,'Position');
set(gcf,'PaperPositionMode','Auto','PaperUnits','points','PaperSize',[pos(3), pos(4)])
filename = 'lifetime_new2.pdf'; 
% print(gcf,filename,'-painters','-dpdf')
exportgraphics(gcf,filename,'ContentType','vector')
close(gcf)