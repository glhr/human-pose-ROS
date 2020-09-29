#include <chrono>
#include <iostream>
#include <string>
#include "ros/ros.h"
#include <ros/console.h>
#include "visualization_msgs/MarkerArray.h"
#include "visualization_msgs/Marker.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
#include "geometry_msgs/Point.h"
#include "cv_bridge/cv_bridge.h"
#include <sensor_msgs/image_encodings.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

#include <cubemos/engine.h>
#include <cubemos/skeleton_tracking.h>

#include "samples.h"
using CUBEMOS_SKEL_Buffer_Ptr = std::unique_ptr<CM_SKEL_Buffer, void (*)(CM_SKEL_Buffer *)>;


cv::Scalar const skeletonColor = cv::Scalar(100, 254, 213);
cv::Scalar const jointColor = cv::Scalar(222, 55, 22);

cv_bridge::CvImagePtr cv_ptr;
cv_bridge::CvImagePtr cv_ptr1;
cv_bridge::CvImagePtr depth_image;
// sensor_msgs::ImageConstPtr ImagePtr;

struct cmPoint
{
    float color_pixel[2];
    float point3d[3];
    std::string to_string() const
    {
        char buffer[100];
        int cx = snprintf(buffer, 100, "(%.2f, %.2f, %.2f)", point3d[0], point3d[1], point3d[2]);
        return std::string(buffer);
    }
};

cmPoint point;
    
void imageCb(const sensor_msgs::ImageConstPtr &msg);
void pointcloudCb(const sensor_msgs::ImageConstPtr &msg);
cmPoint get_skeleton_point_3d(int x, int y);
CUBEMOS_SKEL_Buffer_Ptr create_skel_buffer();
inline void renderSkeletons(const CM_SKEL_Buffer *skeletons_buffer, cv::Mat &image, ros::Publisher skeleton_pub, float *r, float *g, float *b);
int run(int argc, char *argv[]);


void pointcloudCb(const sensor_msgs::ImageConstPtr &msg)
{
    // std::cout << "depth" << std::endl;
    depth_image = cv_bridge::toCvCopy(msg, "32FC1");
}

void imageCb(const sensor_msgs::ImageConstPtr &msg)
{

    // ImagePtr = msg;
    // std::cout << "image" << std::endl;
    cv_ptr1 = cv_bridge::toCvCopy(msg, "bgr8");

    // std::cout << cv_ptr->image.step << std::endl;
}

CUBEMOS_SKEL_Buffer_Ptr create_skel_buffer()
{
    return CUBEMOS_SKEL_Buffer_Ptr(new CM_SKEL_Buffer(), [](CM_SKEL_Buffer *pb) {
        cm_skel_release_buffer(pb);
        delete pb;
    });
}

cmPoint get_skeleton_point_3d(int x, int y)
{
    // Get the distance at the given pixel
        float distance = depth_image->image.at<float>(y, x);
        // std::cout << distance << std::endl;
 
        point.color_pixel[0] = static_cast<float>(x);
        point.color_pixel[1] = static_cast<float>(y);

        rs2_intrinsics intr;

        //ros intrinsic
        // for (int i = 0; i < 5; i++)
        // {
        //     intr.coeffs[i] = 0.0;
        // }
        // intr.width = 1280;
        // intr.height = 720;
        // intr.fx = 925.7305297851562;
        // intr.fy = 925.7361450195312;
        // intr.ppx = 640.8016967773438;
        // intr.ppy = 369.9321594238281;
        // rs2_distortion model = RS2_DISTORTION_BROWN_CONRADY;
        // intr.model = model;

        // Intel intr
        for (int i = 0; i < 5; i++){
            intr.coeffs[i] = 0.0;
        }
        intr.width = 1280;
        intr.height = 720;
        intr.fx = 660.351;
        intr.fy = 660.351;
        intr.ppx = 650.521;
        intr.ppy = 344.305;
        rs2_distortion model = RS2_DISTORTION_BROWN_CONRADY;
        intr.model = model;
        rs2_deproject_pixel_to_point(point.point3d, &intr, point.color_pixel, distance/1000);

    return point;
}

/*
Render skeletons and tracking ids on top of the color image
*/
inline void renderSkeletons(const CM_SKEL_Buffer *skeletons_buffer, cv::Mat &image, ros::Publisher skeleton_pub, float *r, float *g, float *b)
{
    CV_Assert(image.type() == CV_8UC3);
    const cv::Point2f absentKeypoint(-1.0f, -1.0f);

    const std::vector<std::pair<int, int>> limbKeypointsIds = {{1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}};

    for (int i = 0; i < skeletons_buffer->numSkeletons; i++)
    {
        CV_Assert(skeletons_buffer->skeletons[i].numKeyPoints == 18);

        int id = skeletons_buffer->skeletons[i].id;
        cv::Point2f keyPointHead(skeletons_buffer->skeletons[i].keypoints_coord_x[0],
                                 skeletons_buffer->skeletons[i].keypoints_coord_y[0]);
        // visualization_msgs::MarkerArray ros_skeleton;
        visualization_msgs::Marker marker;
        visualization_msgs::Marker line_strip;
        line_strip.type = visualization_msgs::Marker::LINE_STRIP;
        line_strip.id = i;
        marker.id = i * 100;
        line_strip.scale.x = 0.01;
        line_strip.scale.y = 0.01;
        marker.scale.x = 0.05;
        marker.scale.y = 0.05;
        // marker.scale.z = 0.05;
        line_strip.color.g = 1.0f;
        line_strip.color.a = 1.0;
        marker.color.r = r[id];
        marker.color.g = g[id];
        marker.color.b = b[id];
        marker.color.a = 0.7;
        marker.ns = line_strip.ns  = "points_and_lines";
        marker.type = 8;
        marker.header.frame_id = line_strip.header.frame_id = "wrist_camera_link";
        marker.header.stamp = line_strip.header.stamp = ros::Time::now();
        marker.action = line_strip.action = visualization_msgs::Marker::ADD;
        marker.lifetime = ros::Duration(0.5);

        marker.pose.orientation.w = line_strip.pose.orientation.w =  1.0;
        for (size_t keypointIdx = 0; keypointIdx < skeletons_buffer->skeletons[i].numKeyPoints; keypointIdx++)
        {
  
            const cv::Point2f keyPoint(skeletons_buffer->skeletons[i].keypoints_coord_x[keypointIdx], skeletons_buffer->skeletons[i].keypoints_coord_y[keypointIdx]);
            if (keyPoint != absentKeypoint)
            {
                cv::circle(image, keyPoint, 4, jointColor, -1);
                // get the 3d point and render it on the joints
                cmPoint point3d = get_skeleton_point_3d(static_cast<int>(keyPoint.x), static_cast<int>(keyPoint.y));


                geometry_msgs::Point ros_point;
                ros_point.x = point3d.point3d[0];
                ros_point.y = point3d.point3d[1];
                ros_point.z = point3d.point3d[2];

                marker.points.push_back(ros_point);
                line_strip.points.push_back(ros_point);
            }
            skeleton_pub.publish(marker);
            skeleton_pub.publish(line_strip);
        }

        for (const auto &limbKeypointsId : limbKeypointsIds)
        {
            const cv::Point2f keyPointFirst(skeletons_buffer->skeletons[i].keypoints_coord_x[limbKeypointsId.first],
                                            skeletons_buffer->skeletons[i].keypoints_coord_y[limbKeypointsId.first]);

            const cv::Point2f keyPointSecond(skeletons_buffer->skeletons[i].keypoints_coord_x[limbKeypointsId.second],
                                             skeletons_buffer->skeletons[i].keypoints_coord_y[limbKeypointsId.second]);

            if (keyPointFirst == absentKeypoint || keyPointSecond == absentKeypoint)
            {
                continue;
            }

            cv::line(image, keyPointFirst, keyPointSecond, skeletonColor, 2, cv::LINE_AA);
        }
        for (size_t keypointIdx = 0; keypointIdx < skeletons_buffer->skeletons[i].numKeyPoints; keypointIdx++)
        {
            const cv::Point2f keyPoint(skeletons_buffer->skeletons[i].keypoints_coord_x[keypointIdx],
                                       skeletons_buffer->skeletons[i].keypoints_coord_y[keypointIdx]);
            if (keyPoint != absentKeypoint)
            {
                // found a valid keypoint and displaying the skeleton tracking id next to it
                cv::putText(image,
                            (std::to_string(id)),
                            cv::Point2f(keyPoint.x, keyPoint.y - 20),
                            cv::FONT_HERSHEY_COMPLEX,
                            2,
                            skeletonColor);
                break;
            }
        }
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "Skeleton_talker");
    ros::NodeHandle n;

    ros::Subscriber camera_sub = n.subscribe<sensor_msgs::Image>("/wrist_camera/camera/color/image_raw", 1, imageCb);
    ros::Subscriber depth_sub = n.subscribe<sensor_msgs::Image>("/wrist_camera/camera/aligned_depth_to_color/image_raw", 1, pointcloudCb);
    ros::Publisher skeleton_pub = n.advertise<visualization_msgs::Marker>("spawn_skeleton", 1);

    int num_colors = 100;
    float color_r[num_colors];
    float color_g[num_colors];
    float color_b[num_colors];
    srand(100);
    for (int i = 0; i < num_colors; i++)
    {
        color_r[i] = (float)(rand() % 100) / 100;
        color_g[i] = (float)(rand() % 100) / 100;
        color_b[i] = (float)(rand() % 100) / 100;
    }

    CM_TargetComputeDevice enInferenceMode = CM_TargetComputeDevice::CM_CPU;

    if (argc > 1)
    {
        if (strcmp(argv[1], "CPU") == 0 || strcmp(argv[1], "MYRIAD") == 0 || strcmp(argv[1], "GPU") == 0)
        {
            if (strcmp(argv[1], "MYRIAD") == 0)
                enInferenceMode = CM_TargetComputeDevice::CM_MYRIAD;
            else if (strcmp(argv[1], "CPU") == 0)
                enInferenceMode = CM_TargetComputeDevice::CM_CPU;
            else if (strcmp(argv[1], "GPU") == 0)
                enInferenceMode = CM_TargetComputeDevice::CM_GPU;
        }
    }

    CM_SKEL_Handle *handle = nullptr;
    // Output all messages with severity level INFO or higher to the console
    cm_initialise_logging(CM_LogLevel::CM_LL_INFO, true, default_log_dir().c_str());

    CM_ReturnCode retCode = cm_skel_create_handle(&handle, default_license_dir().c_str());
    CHECK_HANDLE_CREATION(retCode);

    // Initialise cubemos DNN framework with the required model
    std::string modelName = default_model_dir();
    if (enInferenceMode == CM_TargetComputeDevice::CM_CPU)
    {
        modelName += std::string("/fp32/skeleton-tracking.cubemos");
    }
    else
    {
        modelName += std::string("/fp16/skeleton-tracking.cubemos");
    }
    retCode = cm_skel_load_model(handle, enInferenceMode, modelName.c_str());
    if (retCode != CM_SUCCESS)
    {
        EXIT_PROGRAM("Model loading failed.");
    }

    const std::string cvWindowName = "cubemos: skeleton tracking C/C++";
    cv::namedWindow(cvWindowName, cv::WINDOW_NORMAL);
    cv::setWindowProperty(cvWindowName, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    const int nHeight = 192; // height of the image with which the DNN model will run inference

    // create an async request handle
    CM_SKEL_AsyncRequestHandle *skeletRequestHandle = nullptr;
    cm_skel_create_async_request_handle(handle, &skeletRequestHandle);

    boost::shared_ptr<sensor_msgs::Image const> ros_image(new sensor_msgs::Image);

    ros_image = ros::topic::waitForMessage<sensor_msgs::Image>("/wrist_camera/camera/color/image_raw", n);

    sensor_msgs::Image ros_img;

    ros_img = *ros_image;

    cv_ptr = cv_bridge::toCvCopy(ros_img, "bgr8");

    CM_Image imageLast = {
        cv_ptr->image.data, CM_UINT8, cv_ptr->image.cols, cv_ptr->image.rows, cv_ptr->image.channels(),
        (int)cv_ptr->image.step[0], CM_HWC};

    CUBEMOS_SKEL_Buffer_Ptr skeletonsPresent = create_skel_buffer();
    CUBEMOS_SKEL_Buffer_Ptr skeletonsLast = create_skel_buffer();
    int nTimeoutMs = 1000;
    // Send the first inference request
    CM_ReturnCode retCodeFirstFrame =
        cm_skel_estimate_keypoints_start_async(handle, skeletRequestHandle, &imageLast, nHeight);
    // Wait until the first results are available
    // Get the skeleton keypoints for the first frame
    retCodeFirstFrame = cm_skel_wait_for_keypoints(handle, skeletRequestHandle, skeletonsLast.get(), nTimeoutMs);

    // continue to loop through acquisition and display until the escape key is hit
    int frameCount = 0;
    std::string fpsTest = "Frame rate: ";

    // start measuring the time taken for execution
    std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();

    // continue to loop through acquisition and display until the escape key is hit
    while (cv::waitKey(1) != 27)
    {
        if (cv_ptr1 && depth_image)
        {
            CM_Image imagePresent = {
                cv_ptr1->image.data, CM_UINT8, cv_ptr1->image.cols, cv_ptr1->image.rows, cv_ptr1->image.channels(),
                (int)cv_ptr1->image.step[0], CM_HWC};
            

            // Run Skeleton Tracking and display the results
            retCode = cm_skel_estimate_keypoints_start_async(handle, skeletRequestHandle, &imagePresent, nHeight);
            retCode = cm_skel_wait_for_keypoints(handle, skeletRequestHandle, skeletonsPresent.get(), nTimeoutMs);
            // track the skeletons in case of successful skeleton estimation
            if (retCode == CM_SUCCESS)
            {
                if (skeletonsPresent->numSkeletons > 0)
                {
                    // Assign tracking ids to the skeletons in the present frame
                    cm_skel_update_tracking_id(handle, skeletonsLast.get(), skeletonsPresent.get());
                    // Render skeleton overlays with tracking ids
                    renderSkeletons(skeletonsPresent.get(), cv_ptr1->image, skeleton_pub, color_r, color_g, color_b);
                    // Set the present frame as last one to track the next frame
                    skeletonsLast.swap(skeletonsPresent);
                    // Free memory of the latest frame
                    cm_skel_release_buffer(skeletonsPresent.get());
                }
            }

            frameCount++;
            if (frameCount % 25 == 0)
            {
                auto timePassed =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime)
                        .count();
                auto fps = 25000.0 / timePassed;

                fpsTest = "Frame rate: " + std::to_string(fps) + " FPS";
                startTime = std::chrono::system_clock::now();
            }

            cv::putText(cv_ptr1->image, fpsTest, cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 1, skeletonColor);
            cv::imshow(cvWindowName, cv_ptr1->image);
            cv_ptr1.reset();
            depth_image.reset();
        }
        ros::spinOnce();
    }
    // release the memory which is managed by the cubemos framework
    cm_skel_destroy_async_request_handle(&skeletRequestHandle);
    cm_skel_destroy_handle(&handle);

    return 0;
}
