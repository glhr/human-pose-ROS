cmake_minimum_required(VERSION 2.8.3)
project(human_pose_ROS LANGUAGES CXX)

# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS rospy roscpp message_generation message_runtime geometry_msgs)
find_package(CUBEMOS_SKELETON_TRACKING EXACT REQUIRED PATHS /opt/cubemos/skeleton_tracking/lib/cmake/CUBEMOS_SKELETON_TRACKING)
find_package(OpenCV REQUIRED CONFIG)
find_package(realsense2)

include_directories(${catkin_INCLUDE_DIRS})
## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

catkin_python_setup()


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES CCN
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

add_executable(cpp-realsense src/intel_pose/cpp_realsense.cpp src/intel_pose/samples.h)
target_link_libraries(cpp-realsense ${catkin_LIBRARIES})
target_link_libraries(cpp-realsense PRIVATE cubemos_skeleton_tracking)
target_link_libraries(cpp-realsense PRIVATE ${OpenCV_LIBS})
target_link_libraries(cpp-realsense PRIVATE ${realsense2_LIBRARY})



set_target_properties (cpp-realsense PROPERTIES
	FOLDER samples
)

# install(TARGETS cpp-realsense
# 	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# )

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
