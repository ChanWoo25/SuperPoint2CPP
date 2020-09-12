# cmake_minimum_required(VERSION 3.0)
# project(CreateVaca
# VERSION     1.0.0
# LANGUAGES   CXX
# DESCRIPTION "Create Vocabulary file using Superpoint features." 

option(BUILD_DBoW2   "Build DBoW2"            ON)
option(BUILD_Test    "Build test application" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# set(HDRS
#   include/BowVector.h           
#   include/QueryResults.h        include/TemplatedDatabase.h   
#   include/FClass.h              include/FeatureVector.h
#   include/ScoringObject.h       include/TemplatedVocabulary.h include/FSuperpoint.h)
set(SRCS 
  src/BowVector.cpp     src/FSuperpoint.cpp
  src/FeatureVector.cpp src/QueryResults.cpp  src/ScoringObject.cpp)

# set(DEPENDENCY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
# set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)

### Root CMakeLists.txt already find OpenCV.
# find_package(OpenCV 3.4.11 REQUIRED)

message("CreateVoca's home directory is ${CMAKE_HOME_DIRECTORY}")

add_library(DBoW2 SHARED ${SRCS})
target_include_directories(DBoW2 PUBLIC 
    ${CMAKE_HOME_DIRECTORY}/CreateVoca/include 
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(DBoW2 ${OpenCV_LIBS})
set_target_properties(DBoW2 PROPERTIES 
CXX_STANDARD 14
LIBRARY_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/lib"
)

# if(BUILD_Test)
#   add_executable(test test/test.cpp)
#   target_link_libraries(test ${PROJECT_NAME} ${OpenCV_LIBS})
#   set_target_properties(test PROPERTIES CXX_STANDARD 11)
# endif(BUILD_Test)



# configure_file(src/DBoW2.cmake.in
#   "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" @ONLY)

# install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
# if(BUILD_DBoW2)
#   install(DIRECTORY include/DBoW2 DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
# endif()
# install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DBoW2Config.cmake"
#   DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
# install(FILES "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" 
#   DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/DBoW2/)
# install(DIRECTORY ${DEPENDENCY_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX} OPTIONAL)

