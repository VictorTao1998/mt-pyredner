cmake_minimum_required(VERSION 3.5)
project(AR_DEPTH)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
SET(CMAKE_CXX_FLAGS  "-O2")

add_executable(densify src/main.cpp)
target_link_libraries (densify ${OpenCV_LIBS})
