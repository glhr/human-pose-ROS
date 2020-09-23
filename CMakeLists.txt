cmake_minimum_required(VERSION 2.8.3)
project(human_pose_ROS LANGUAGES CXX)

# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS rospy roscpp message_generation message_runtime geometry_msgs)
find_package(CUBEMOS_SKELETON_TRACKING REQUIRED PATHS /opt/cubemos/skeleton_tracking/lib/cmake/CUBEMOS_SKELETON_TRACKING)


set(OpenCV_DIR "/opt/cubemos/skeleton_tracking/samples/dep/opencv/lib/cmake/opencv4")

find_package(OpenCV REQUIRED CONFIG)

find_package(realsense2 REQUIRED CONFIG PATHS "/opt/cubemos/skeleton_tracking/samples/dep/realsense/lib/cmake/realsense2")

include (src/intel_pose/cmake/linux_compiler_flags.cmake)
# FIND_PACKAGE(PkgConfig REQUIRED)
# PKG_CHECK_MODULES(GTK2 REQUIRED gtk+-2.0)

# # Setup CMake to use GTK+, tell the compiler where to look for headers
# # and to the linker where to look for libraries
# INCLUDE_DIRECTORIES(${GTK2_INCLUDE_DIRS})
# LINK_DIRECTORIES(${GTK2_LIBRARY_DIRS})

# # Add other flags to the compiler
# ADD_DEFINITIONS(${GTK2_CFLAGS_OTHER})

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
include(GNUInstallDirs)



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
# target_link_libraries(cpp-realsense ${GTK2_LIBRARIES})



set_target_properties (cpp-realsense PROPERTIES
	FOLDER samples
)



# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

install(TARGETS cpp-realsense
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
