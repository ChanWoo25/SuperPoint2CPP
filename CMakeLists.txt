cmake_minimum_required(VERSION 3.12...3.18)

project(
SuperPointCpp   
VERSION         20.08
LANGUAGES       CXX
DESCRIPTION     "C++ Implementation of Superpoint."
)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/libtorch" "/usr/local/zed")
find_package(Torch REQUIRED QUIET)
find_package(OpenCV 3.0 QUIET)
find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)

include_directories(${ZED_INCLUDE_DIRS})

if(OpenCV_FOUND)
    message(STATUS "OpenCV >= 3.0 found.")
else()
    message(STATUS "OpenCV cannot be found.")
endif()

add_subdirectory(src)
#add_subdirectory(zed)

add_executable(main test_main.cpp)
target_compile_features(main PUBLIC cxx_std_14)
target_link_libraries(main PUBLIC 
superpoint
#libzed
${TORCH_LIBRARIES}
${OpenCV_LIBS}
)
set_target_properties(main PROPERTIES
RUNTIME_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/bin"
)


# target_include_directories(main
# PUBLIC  ${PROJECT_SOURCE_DIR}/include
# )