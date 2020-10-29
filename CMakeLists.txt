cmake_minimum_required(VERSION 2.8.3)
project(human_pose_ROS LANGUAGES CXX)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS rospy roscpp message_generation message_runtime geometry_msgs sensor_msgs)

include_directories(${catkin_INCLUDE_DIRS})


catkin_python_setup()

add_message_files(
  FILES
	Skeleton.msg
  PoseEstimation.msg
  Limbs.msg
)

generate_messages(
	DEPENDENCIES
	std_msgs
	sensor_msgs
	geometry_msgs  # Or other packages containing msgs
)

include(${CMAKE_CURRENT_SOURCE_DIR}/src/intel_pose/CMakeLists.txt OPTIONAL)
