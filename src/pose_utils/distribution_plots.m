clear all
close all
time = ["1606816796.9493315","1606818486.7909591","1606826033.379091"]
distances = [1,2,3]
joints = ["nose" "left_eye" "right_eye" "left_ear" "left_shoulder" "right_shoulder" "left_elbow" "left_wrist" "left_hip" "right_hip" "left_knee" "right_knee" "right_ankle"]
folder = "/home/slave/Documents/workspaces/real_ws/src/lh7-nlp/human_pose_ROS/src/pose_utils/joint_data/"


for d = 1:length(distances)
    joints_valid = [];
    total_x = [];
    total_y = [];
    total_z = [];

    for i =1:length(joints)
        try
            path = strcat(folder,time(d),"-",joints(i),".csv");
            data = readtable(path);

            x = table2array(data(:,1));
            y = table2array(data(:,2));
            z = table2array(data(:,3));

            x = normalize(x,'center','median');
            y = normalize(y,'center','median');
            z = normalize(z,'center','median');

            joints_valid = [joints_valid; joints(i)];
            total_x = [total_x; x];
            total_y = [total_y; y];
            total_z = [total_z; z];

    %         subplot(1,2,1);
    %         hist3([x,y],"Nbins", [20,20],"CDataMode","auto","FaceColor","interp")
    %         xlabel("X (camera frame)")
    %         ylabel("Y (camera frame)")
    %     
    %         subplot(1,2,2);
    %         hist3([x,z],"Nbins", [20,20],"CDataMode","auto","FaceColor","interp")
    %         xlabel("X (camera frame)")
    %         ylabel("Z (camera frame)")
    %         fig = strcat(time,"-",joints(i),".fig")
    %         savefig(fig)
    %         close all
        catch
            warning("Failed to open csv, skipping")
        end
    end
    if d == 1
        total_1 = [total_x total_y total_z];
    end
    if d == 2
        total_2 = [total_x total_y total_z];
    end
    if d == 3
        total_3 = [total_x total_y total_z];
    end
end

totals_x = [total_1(:,1); total_2(:,1); total_3(:,1)]
totals_y = [total_1(:,2); total_2(:,2); total_3(:,2)];
totals_z = [total_1(:,3); total_2(:,3); total_3(:,3)];

g = [zeros(length(total_1), 1); ones(length(total_2), 1); 2*ones(length(total_3), 1)];
g = [repelem(["1 meter", "2 meters", "3 meters"], [length(total_1), length(total_2), length(total_3)])];

figure(1)
boxplot(totals_x, g,'Whisker',1,'Symbol','k+')
xlabel('Person distance from robot (m)')
ylabel('Normalized joint positions (m)')
ylim([-0.15 0.16]);

colors = [[114, 178, 255]; [39, 174, 96]; [240, 178, 122]];
colors = colors./255;
h = findobj(gca,'Tag','Box');
for j=1:length(h)
    patch(get(h(j),'XData'),get(h(j),'YData'),colors(j,:),'FaceAlpha',.5);
end
set(gcf,'position',[0,0,400,600])

figure(2)
boxplot(totals_y, g,'Whisker',1,'Symbol','k+')
xlabel('Person distance from robot (m)')
ylabel('Normalized joint positions (m)')
ylim([-0.15 0.16]);

h = findobj(gca,'Tag','Box');
for j=1:length(h)
    patch(get(h(j),'XData'),get(h(j),'YData'),colors(j,:),'FaceAlpha',.5);
end
set(gcf,'position',[0,0,400,600])

figure(3)
boxplot(totals_z, g,'Whisker',1,'Symbol','k+')
xlabel('Person distance from robot (m)')
ylabel('Normalized joint positions (m)')
ylim([-0.15 0.16]);

h = findobj(gca,'Tag','Box');
for j=1:length(h)
    patch(get(h(j),'XData'),get(h(j),'YData'),colors(j,:),'FaceAlpha',.5);
end
set(gcf,'position',[0,0,400,600])
% T = table(joints_valid,std_x,std_y,std_z)
