cmake_minimum_required(VERSION 3.0.2)
project(teamrobo2026)

## Find catkin and required packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    visualization_msgs
    message_generation
    tf
    tf2
    tf2_geometry_msgs
    tf_conversions
    laser_geometry
    pcl_ros
    pcl_conversions
    obstacle_detector
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
    
    # ACADOS include directories
    set(ACADOS_INCLUDE_DIRS
        ${ACADOS_INSTALL_DIR}/include
        ${ACADOS_INSTALL_DIR}/include/blasfeo/include
        ${ACADOS_INSTALL_DIR}/include/hpipm/include
        ${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code
    )
    
    # ACADOS library directories
    set(ACADOS_LIBRARY_DIRS
        ${ACADOS_INSTALL_DIR}/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code
    )
    
    # ACADOS libraries
    set(ACADOS_LIBRARIES
        acados
        blasfeo
        hpipm
    )
endif()

###################################
## catkin specific configuration ##
###################################
catkin_package(
    CATKIN_DEPENDS 
        roscpp 
        rospy
        std_msgs 
        sensor_msgs 
        geometry_msgs 
        nav_msgs
        visualization_msgs
        message_runtime
        tf
        tf2
        tf2_geometry_msgs
        tf_conversions
        laser_geometry
        pcl_ros
        pcl_conversions
        obstacle_detector   
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${GAZEBO_INCLUDE_DIRS}
    ${ACADOS_INCLUDE_DIRS}
)

link_directories(
    ${GAZEBO_LIBRARY_DIRS}
    ${PCL_LIBRARY_DIRS}
    ${ACADOS_LIBRARY_DIRS}
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
add_dependencies(inspection_node 
    ${${PROJECT_NAME}_EXPORTED_TARGETS} 
    ${catkin_EXPORTED_TARGETS}
)

# Laser scan to point cloud node
add_executable(laser_scan_to_point_cloud_node 
    src/laser_scan_to_point_cloud.cpp
)
target_link_libraries(laser_scan_to_point_cloud_node
    ${catkin_LIBRARIES}
    ${EIGEN3_LIBRARIES}
)
add_dependencies(laser_scan_to_point_cloud_node 
    ${${PROJECT_NAME}_EXPORTED_TARGETS} 
    ${catkin_EXPORTED_TARGETS}
)

# Occupancy grid to point cloud node
add_executable(occupancy_to_cloud_node 
    src/occupancy_to_cloud.cpp
)
target_link_libraries(occupancy_to_cloud_node
    ${catkin_LIBRARIES}
)
add_dependencies(occupancy_to_cloud_node 
    ${${PROJECT_NAME}_EXPORTED_TARGETS} 
    ${catkin_EXPORTED_TARGETS}
)

# MPC node with ACADOS
if(ACADOS_INSTALL_DIR)
    add_executable(mpc_node 
        src/mpc_node.cpp
    )
    
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code/acados_solver_jackal_diff_drive.c")
        # Build the ACADOS solver library from source
        add_library(acados_solver_lib STATIC
            c_generated_code/acados_solver_jackal_diff_drive.c
            c_generated_code/acados_sim_solver_jackal_diff_drive.c
            c_generated_code/jackal_diff_drive_model/jackal_diff_drive_expl_ode_fun.c
            c_generated_code/jackal_diff_drive_model/jackal_diff_drive_expl_vde_forw.c
            c_generated_code/jackal_diff_drive_model/jackal_diff_drive_expl_vde_adj.c
            c_generated_code/jackal_diff_drive_constraints/jackal_diff_drive_constr_h_fun.c
            c_generated_code/jackal_diff_drive_constraints/jackal_diff_drive_constr_h_fun_jac_uxt_zt.c
        )
        target_include_directories(acados_solver_lib PUBLIC
            ${ACADOS_INCLUDE_DIRS}
        )
        
        target_link_libraries(mpc_node
            acados_solver_lib
            ${catkin_LIBRARIES}
            ${ACADOS_LIBRARIES}
            m  # Math library
        )
    else()
        message(WARNING "ACADOS generated code not found. Please run generate_acados_solver.py first.")
        target_link_libraries(mpc_node
            ${catkin_LIBRARIES}
            ${ACADOS_LIBRARIES}
            m  # Math library
        )
    endif()
    
    add_dependencies(mpc_node 
        ${${PROJECT_NAME}_EXPORTED_TARGETS} 
        ${catkin_EXPORTED_TARGETS}
    )
else()
    message(WARNING "ACADOS not configured. MPC node will not be built.")
endif()

catkin_install_python(PROGRAMS 
  script/generate_acados_solver.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

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
