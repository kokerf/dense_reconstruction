cmake_minimum_required(VERSION 3.8)
project(dense_reconstruction)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_STANDARD 11)

# OpenCV
find_package(OpenCV 3.1.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})

# add exe
include_directories(${CMAKE_SOURCE_DIR}/src)

set(SOURCE_FILES
        main.cpp
        src/frame.cpp
        src/depth_filter.cpp
        src/viewer.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES})