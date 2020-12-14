cmake_minimum_required(VERSION 3.2)

project(
SuperPointCpp   
VERSION         20.08
LANGUAGES       CXX
DESCRIPTION     "C++ Implementation of Superpoint."
)

set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)

### LIBTORCH
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/libtorch")
find_package(Torch REQUIRED QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})
###

### OPENCV
find_package(OpenCV 3.4.11 QUIET)
message(STATUS "Found OpenCV version is ${OpenCV_VERSION}")
message(STATUS "Found OpenCV include is ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found OpenCV libraries is ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})
###

# ### ZED & CUDA
# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/zed")
# find_package(ZED 3 REQUIRED)
# find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
# include_directories(${ZED_INCLUDE_DIRS})
# ###

### SUB_DIRECTORIES
add_subdirectory(src)
add_subdirectory(CreateVoca)
# add_subdirectory(zed)

# ### Deprecated ###
# add_executable(main main_SuperPoint.cpp)
# target_compile_features(main PUBLIC cxx_std_14)
# target_link_libraries(main PUBLIC 
# superpoint
# #libzed
# ${TORCH_LIBRARIES}
# ${OpenCV_LIBS}
# )
# set_target_properties(main PROPERTIES
# RUNTIME_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/bin"
# )

### Runfile to test SuperPoint
add_executable(SuperPoint main_SuperPoint.cpp)
target_compile_features(SuperPoint PUBLIC cxx_std_14)
target_link_libraries(SuperPoint PUBLIC 
Super
${TORCH_LIBRARIES}
${OpenCV_LIBS}
)
set_target_properties(SuperPoint PROPERTIES
RUNTIME_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/bin"
)

### Runfile to create Vocabulary
add_executable(CreateVoca main_CreateVoca.cpp)
target_link_libraries(CreateVoca PUBLIC 
Super
DBoW2
# ${TORCH_LIBRARIES}
# ${OpenCV_LIBS}
)
set_target_properties(CreateVoca PROPERTIES
CXX_STANDARD 11
RUNTIME_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/bin"
)