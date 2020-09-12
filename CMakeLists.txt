cmake_minimum_required(VERSION 3.12...3.18)

project(
SuperPointCpp   
VERSION         20.08
LANGUAGES       CXX
DESCRIPTION     "C++ Implementation of Superpoint."
)

set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)

### LIBTORCH
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/libtorch" "/usr/local/zed")
find_package(Torch REQUIRED QUIET)
include_directories(${TORCH_INCLUDE_DIRS})
###

### OPENCV
find_package(OpenCV 3.4.11 QUIET)
message(STATUS "Found OpenCV version is ${OpenCV_VERSION}")
message(STATUS "Found OpenCV include is ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found OpenCV libraries is ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})
###

### ZED & CUDA
find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
include_directories(${ZED_INCLUDE_DIRS})
###

### SUB_DIRECTORIES
add_subdirectory(src)
add_subdirectory(CreateVoca)
#add_subdirectory(zed)

###
add_executable(main main_SuperPoint.cpp)
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

### Runfile to test new Framework.
add_executable(Test main.cpp)
target_compile_features(Test PUBLIC cxx_std_14)
target_link_libraries(Test PUBLIC 
Super
${TORCH_LIBRARIES}
${OpenCV_LIBS}
)
set_target_properties(Test PROPERTIES
RUNTIME_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/bin"
)

### Runfile to create Vocabulary
add_executable(CreateVoca main_CreateVoca.cpp)
target_link_libraries(CreateVoca PUBLIC 
superpoint
DBoW2
${TORCH_LIBRARIES}
${OpenCV_LIBS}
)
target_include_directories(CreateVoca PUBLIC 
"${CMAKE_HOME_DIRECTORY}/CreateVoca/include"
)
set_target_properties(CreateVoca PROPERTIES
CXX_STANDARD 14
RUNTIME_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/bin"
)


# target_include_directories(main
# PUBLIC  ${PROJECT_SOURCE_DIR}/include
# )