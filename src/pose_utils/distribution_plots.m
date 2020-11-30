clear all
time = "1606765858.5468237"
joints = ["nose" "left_eye" "right_eye" "left_ear" "left_shoulder" "right_shoulder" "left_elbow" "left_wrist" "left_hip" "right_hip" "left_knee" "right_knee" "right_ankle"]
folder = "/home/slave/Documents/workspaces/real_ws/src/lh7-nlp/human_pose_ROS/src/pose_utils/joint_data/"

joints_valid = [];
std_x = [];
std_y = [];
std_z = [];

for i =1:length(joints);
    try
        path = strcat(folder,time,"-",joints(i),".csv");
        data = readtable(path);

        x = table2array(data(:,1));
        y = table2array(data(:,2));
        z = table2array(data(:,3));

        x = normalize(x,'center');
        y = normalize(y,'center');
        z = normalize(z,'center');

        joints_valid = [joints_valid; joints(i)];
        std_x = [std_x; std(x)];
        std_y = [std_y; std(y)];
        std_z = [std_z; std(z)];

        subplot(1,2,1);
        hist3([x,y],"Nbins", [20,20],"CDataMode","auto","FaceColor","interp")
        xlabel("X (camera frame)")
        ylabel("Y (camera frame)")
    
        subplot(1,2,2);
        hist3([x,z],"Nbins", [20,20],"CDataMode","auto","FaceColor","interp")
        xlabel("X (camera frame)")
        ylabel("Z (camera frame)")
        fig = strcat(time,"-",joints(i),".fig")
        savefig(fig)
        close all
    catch
        warning("Failed to open csv, skipping")
    end
end;


T = table(joints_valid,std_x,std_y,std_z)