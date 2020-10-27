#include <chrono>
#include <iostream>
#include <string>
#include "ros/ros.h"
#include <ros/console.h>
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include <sensor_msgs/image_encodings.h>



#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cubemos/engine.h>
#include <cubemos/skeleton_tracking.h>

#include "samples.h"

using CUBEMOS_SKEL_Buffer_Ptr = std::unique_ptr<CM_SKEL_Buffer, void (*)(CM_SKEL_Buffer*)>;
static cv::Scalar const skeletonColor = cv::Scalar(100, 254, 213);
static cv::Scalar const jointColor = cv::Scalar(222, 55, 22);

cv_bridge::CvImagePtr cv_ptr;
cv_bridge::CvImagePtr cv_ptr1;

// void imageCb(const sensor_msgs::ImageConstPtr& msg){
    
    
//     cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
//     // std::cout << cv_ptr->image.step << std::endl;




// }

CUBEMOS_SKEL_Buffer_Ptr create_skel_buffer(){
    return CUBEMOS_SKEL_Buffer_Ptr(new CM_SKEL_Buffer(), [](CM_SKEL_Buffer* pb) {
        cm_skel_release_buffer(pb);
        delete pb;
    });
}

/*
Render skeletons and tracking ids on top of the color image
*/
inline void renderSkeletons(const CM_SKEL_Buffer* skeletons_buffer, cv::Mat& image){
    CV_Assert(image.type() == CV_8UC3);
    const cv::Point2f absentKeypoint(-1.0f, -1.0f);
    
    ros::spinOnce();
    const std::vector<std::pair<int, int>> limbKeypointsIds = { { 1, 2 },   { 1, 5 },   { 2, 3 }, { 3, 4 },  { 5, 6 },
                                                                { 6, 7 },   { 1, 8 },   { 8, 9 }, { 9, 10 }, { 1, 11 },
                                                                { 11, 12 }, { 12, 13 }, { 1, 0 }, { 0, 14 }, { 14, 16 },
                                                                { 0, 15 },  { 15, 17 } };

    for (int i = 0; i < skeletons_buffer->numSkeletons; i++) {
        CV_Assert(skeletons_buffer->skeletons[i].numKeyPoints == 18);

        int id = skeletons_buffer->skeletons[i].id;
        cv::Point2f keyPointHead(skeletons_buffer->skeletons[i].keypoints_coord_x[0],
                                 skeletons_buffer->skeletons[i].keypoints_coord_y[0]);

        for (size_t keypointIdx = 0; keypointIdx < skeletons_buffer->skeletons[i].numKeyPoints; keypointIdx++) {
            const cv::Point2f keyPoint(skeletons_buffer->skeletons[i].keypoints_coord_x[keypointIdx],
                                       skeletons_buffer->skeletons[i].keypoints_coord_y[keypointIdx]);
            if (keyPoint != absentKeypoint) {
                cv::circle(image, keyPoint, 4, jointColor, -1);
            }
        }

        for (const auto& limbKeypointsId : limbKeypointsIds) {
            const cv::Point2f keyPointFirst(skeletons_buffer->skeletons[i].keypoints_coord_x[limbKeypointsId.first],
                                            skeletons_buffer->skeletons[i].keypoints_coord_y[limbKeypointsId.first]);

            const cv::Point2f keyPointSecond(skeletons_buffer->skeletons[i].keypoints_coord_x[limbKeypointsId.second],
                                             skeletons_buffer->skeletons[i].keypoints_coord_y[limbKeypointsId.second]);

            if (keyPointFirst == absentKeypoint || keyPointSecond == absentKeypoint) {
                continue;
            }

            cv::line(image, keyPointFirst, keyPointSecond, skeletonColor, 2, cv::LINE_AA);
        }
        for (size_t keypointIdx = 0; keypointIdx < skeletons_buffer->skeletons[i].numKeyPoints; keypointIdx++) {
            const cv::Point2f keyPoint(skeletons_buffer->skeletons[i].keypoints_coord_x[keypointIdx],
                                       skeletons_buffer->skeletons[i].keypoints_coord_y[keypointIdx]);
            if (keyPoint != absentKeypoint) {
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

int main(int argc, char* argv[]){
    ros::init(argc, argv, "Skeleton_talker");
    ros::NodeHandle n;

    // ros::Subscriber camera_sub = n.subscribe<sensor_msgs::Image>("/wrist_camera/camera/color/image_raw", 1, imageCb);

    CM_TargetComputeDevice enInferenceMode = CM_TargetComputeDevice::CM_CPU;

    std::string szInput = "webcam";
    if (argc > 1) {
        if (strcmp(argv[1], "CPU") == 0 || strcmp(argv[1], "MYRIAD") == 0 || strcmp(argv[1], "GPU") == 0) {
            if (strcmp(argv[1], "MYRIAD") == 0)
                enInferenceMode = CM_TargetComputeDevice::CM_MYRIAD;
            else if (strcmp(argv[1], "CPU") == 0)
                enInferenceMode = CM_TargetComputeDevice::CM_CPU;
            else if (strcmp(argv[1], "GPU") == 0)
                enInferenceMode = CM_TargetComputeDevice::CM_GPU;
        }
    }
    if (argc > 2) {
        if (strcmp(argv[2], "webcam") != 0) {
            szInput = argv[2];
        }
    }

    CM_SKEL_Handle* handle = nullptr;
    // Output all messages with severity level INFO or higher to the console
    cm_initialise_logging(CM_LogLevel::CM_LL_INFO, true, default_log_dir().c_str());

    CM_ReturnCode retCode = cm_skel_create_handle(&handle, default_license_dir().c_str());
    CHECK_HANDLE_CREATION(retCode);

    // Initialise cubemos DNN framework with the required model
    std::string modelName = default_model_dir();
    if (enInferenceMode == CM_TargetComputeDevice::CM_CPU) {
        modelName += std::string("/fp32/skeleton-tracking.cubemos");
    }
    else {
        modelName += std::string("/fp16/skeleton-tracking.cubemos");
    }
    retCode = cm_skel_load_model(handle, enInferenceMode, modelName.c_str());
    if (retCode != CM_SUCCESS) {
        EXIT_PROGRAM("Model loading failed.");
    }

    const std::string cvWindowName = "cubemos: skeleton tracking C/C++";
    cv::namedWindow(cvWindowName, cv::WINDOW_NORMAL);
    cv::setWindowProperty(cvWindowName, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    // cv::VideoCapture webcam;
    // if (szInput == "webcam")
    //     webcam.open(0);
    // else
    //     webcam.open(szInput);

    // cv::Mat capturedFrame;

    const int nHeight = 192; // height of the image with which the DNN model will run inference

    // create an async request handle
    CM_SKEL_AsyncRequestHandle* skeletRequestHandle = nullptr;
    cm_skel_create_async_request_handle(handle, &skeletRequestHandle);

    // cache the first inference to get started with tracking
    // webcam.read(capturedFrame);

    boost::shared_ptr<sensor_msgs::Image const> ros_image(new sensor_msgs::Image);

    ros_image = ros::topic::waitForMessage<sensor_msgs::Image>("/wrist_camera/camera/color/image_raw", n);
    
    sensor_msgs::Image ros_img;

    
    ros_img = *ros_image;
    
    

    cv_ptr = cv_bridge::toCvCopy(ros_img, "bgr8");

    CM_Image imageLast = {
    cv_ptr->image.data,         CM_UINT8, cv_ptr->image.cols , cv_ptr->image.rows, cv_ptr->image.channels(),
    (int)cv_ptr->image.step[0], CM_HWC};

    // CM_Image imageLast = {
    //     capturedFrame.data,         CM_UINT8, capturedFrame.cols, capturedFrame.rows, capturedFrame.channels(),
    //     (int)capturedFrame.step[0], CM_HWC
    // };
    
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
    while (cv::waitKey(1) != 27) {
        // capture image from a webcamera
        // webcam.read(capturedFrame);

        boost::shared_ptr<sensor_msgs::Image const> ros_image1(new sensor_msgs::Image);

        ros_image1 = ros::topic::waitForMessage<sensor_msgs::Image>("/wrist_camera/camera/color/image_raw", n);
        
        sensor_msgs::Image ros_img1;

        
        ros_img1 = *ros_image1;
        
        
        cv_ptr1 = cv_bridge::toCvCopy(ros_img1, "bgr8");
        
        // exit the loop if the captured frame is empty
        // if (capturedFrame.empty()) {
        //     std::cerr << "No new frame could be captured using the input source. Exiting the loop." << std::endl;
        //     break;
        // }

        CM_Image imagePresent = {
        cv_ptr1->image.data,         CM_UINT8, cv_ptr1->image.cols , cv_ptr1->image.rows, cv_ptr1->image.channels(),
        (int)cv_ptr1->image.step[0], CM_HWC};

        // CM_Image imagePresent = {
        //     capturedFrame.data,         CM_UINT8, capturedFrame.cols, capturedFrame.rows, capturedFrame.channels(),
        //     (int)capturedFrame.step[0], CM_HWC
        // };

        // Run Skeleton Tracking and display the results
        retCode = cm_skel_estimate_keypoints_start_async(handle, skeletRequestHandle, &imagePresent, nHeight);
        retCode = cm_skel_wait_for_keypoints(handle, skeletRequestHandle, skeletonsPresent.get(), nTimeoutMs);

        // track the skeletons in case of successful skeleton estimation
        if (retCode == CM_SUCCESS) {
            if (skeletonsPresent->numSkeletons > 0) {
                // Assign tracking ids to the skeletons in the present frame
                cm_skel_update_tracking_id(handle, skeletonsLast.get(), skeletonsPresent.get());
                // Render skeleton overlays with tracking ids
                renderSkeletons(skeletonsPresent.get(), cv_ptr1->image);
                // Set the present frame as last one to track the next frame
                skeletonsLast.swap(skeletonsPresent);
                // Free memory of the latest frame
                cm_skel_release_buffer(skeletonsPresent.get());
            }
        }

	frameCount++;
        if (frameCount % 25 == 0) {
            auto timePassed =
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime)
                .count();
            auto fps = 25000.0 / timePassed;

            fpsTest = "Frame rate: " + std::to_string(fps) + " FPS";
            startTime = std::chrono::system_clock::now();
        }
        cv::putText(cv_ptr1->image, fpsTest, cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 1, skeletonColor);
        cv::imshow(cvWindowName, cv_ptr1->image);
    }

    // release the memory which is managed by the cubemos framework
    cm_skel_destroy_async_request_handle(&skeletRequestHandle);
    cm_skel_destroy_handle(&handle);
    return 0;
}
