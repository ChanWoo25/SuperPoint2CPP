cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SuperPointCpp)


set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/leecw/libtorch")
find_package(Torch REQUIRED)
find_package(OpenCV 3.0 QUIET)
if(OpenCV_FOUND)
   message(STATUS "OpenCV >= 3.0 found.")
endif()


include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
)

add_executable(SP main.cpp src/SuperPoint.cpp)
target_link_libraries(SP PUBLIC 
${TORCH_LIBRARIES}
${OpenCV_LIBS}
)