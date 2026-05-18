cmake_minimum_required(VERSION 3.0.2)
project(teamrobo2026)

set(dependencies
    roscpp
    # rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    visualization_msgs
    tf
    tf2
    tf2_geometry_msgs
    tf_conversions
    laser_geometry
    pcl_ros
    pcl_conversions
    obstacle_detector
)
# Find catkin and required packages
find_package(catkin REQUIRED COMPONENTS
    ${dependencies}
)

## Find system dependencies
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

include(FindPkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(GAZEBO gazebo)
endif()

find_package(Boost REQUIRED COMPONENTS system filesystem regex)
find_package(Protobuf REQUIRED)

## ACADOS configuration
# Set ACADOS_INSTALL_DIR to your ACADOS installation path
# Example: set(ACADOS_INSTALL_DIR "$ENV{HOME}/acados")
set(ACADOS_INSTALL_DIR "$ENV{ACADOS_SOURCE_DIR}" CACHE PATH "ACADOS installation directory")

if(NOT ACADOS_INSTALL_DIR)
    message(WARNING "ACADOS_INSTALL_DIR not set. Set ACADOS_SOURCE_DIR environment variable or specify path in CMakeLists.txt")
else()
    message(STATUS "ACADOS installation directory: ${ACADOS_INSTALL_DIR}")
    
    set(acados_include "${ACADOS_INSTALL_DIR}/include")
    set(acados_lib "${ACADOS_INSTALL_DIR}/lib")
    set(full_model_build "${CMAKE_CURRENT_SOURCE_DIR}/script/c_generated_code")
    
endif()

###################################
## catkin specific configuration ##
###################################
catkin_package(
    CATKIN_DEPENDS ${dependencies}
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
    include # cpp "include" directory
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${acados_include}
    ${acados_include}/blasfeo/include/
    ${acados_include}/hpipm/include/
    ${acados_include}/acados/
    ${acados_include}/qpOASES_e/
    ${full_model_build}
    ${EIGEN3_INCLUDE_DIRS}
    ${GAZEBO_INCLUDE_DIRS}
)

link_directories(
    ${GAZEBO_LIBRARY_DIRS}
    ${PCL_LIBRARY_DIRS}
)

## Add compile definitions
add_definitions(${PCL_DEFINITIONS})

## Declare C++ executables

# Inspection node
add_executable(inspection_node 
    src/inspection_node.cpp
)
target_link_libraries(inspection_node
    ${catkin_LIBRARIES}
)

# Laser scan to point cloud node
add_executable(laser_scan_to_point_cloud_node 
    src/laser_scan_to_point_cloud.cpp
)
target_link_libraries(laser_scan_to_point_cloud_node
    ${catkin_LIBRARIES}
    ${EIGEN3_LIBRARIES}
)

# Occupancy grid to point cloud node
add_executable(occupancy_to_cloud_node 
    src/occupancy_to_cloud.cpp
)
target_link_libraries(occupancy_to_cloud_node
    ${catkin_LIBRARIES}
)

# MPC node with ACADOS
if(ACADOS_INSTALL_DIR)
    add_executable(mpc_node 
        src/mpc_node.cpp
    )
    
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/script/c_generated_code")    
        target_link_libraries(mpc_node
            ${catkin_LIBRARIES}
            ${full_model_build}/libacados_ocp_solver_jackal_diff_drive.so
            ${acados_lib}/libacados.so
        )
    else()
        message(WARNING "ACADOS generated code not found. Please run generate_acados_solver.py first.")
        target_link_libraries(mpc_node
            ${catkin_LIBRARIES}
        )
    endif()
    
else()
    message(WARNING "ACADOS not configured. MPC node will not be built.")
endif()

# catkin_install_python(PROGRAMS 
#   script/generate_acados_solver.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# # Collision publisher node (from original CMakeLists)
# add_executable(collision_publisher_node 
#     src/collision_publisher.cpp
# )
# target_link_libraries(collision_publisher_node
#     ${catkin_LIBRARIES}
#     ${GAZEBO_LIBRARIES} 
#     ${Boost_LIBRARIES} 
#     ${PROTOBUF_LIBRARIES} 
#     pthread
# )

#############
## Install ##
#############

## Mark executables for installation
install(TARGETS 
    inspection_node
    laser_scan_to_point_cloud_node
    occupancy_to_cloud_node
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install MPC node if built
if(TARGET mpc_node)
    install(TARGETS mpc_node
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
endif()
