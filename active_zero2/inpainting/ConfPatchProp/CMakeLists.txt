cmake_minimum_required(VERSION 3.4)
project(confpatchprop)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

file(GLOB SRC_CODE /src/**)

add_library(pmsp ${PROJECT_SOURCE_DIR}/src/pms_propagation.cpp)
add_library(pmsu ${PROJECT_SOURCE_DIR}/src/pms_util.cpp)

add_executable(cpp ${PROJECT_SOURCE_DIR}/src/main.cpp)

target_link_libraries(cpp ${OpenCV_LIBS} pmsp pmsu)
