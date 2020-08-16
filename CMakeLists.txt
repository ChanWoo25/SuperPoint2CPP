cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SuperPointCpp)


set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/leecw/libtorch")
find_package(Torch REQUIRED)
find_package(OpenCV 3.0 QUIET)
if(OpenCV_FOUND)
   message(STATUS "OpenCV >= 3.0 found.")
endif()

message(STATUS ${PROJECT_SOURCE_DIR})
message(STATUS ${CMAKE_SOURCE_DIR})
#message(STATUS ${TORCH_LIBRARIES})

# TEST
add_executable(test test.cpp )
target_include_directories(test
PUBLIC 
   ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test 
${TORCH_LIBRARIES}
${OpenCV_LIBS}
)


# SUPERPOINT
add_executable(SuperPoint main.cpp ./src/SuperPoint.cpp ./src/Extractor.cpp)
target_include_directories(SuperPoint
PUBLIC 
   ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(SuperPoint 
${TORCH_LIBRARIES}
${OpenCV_LIBS}
)
