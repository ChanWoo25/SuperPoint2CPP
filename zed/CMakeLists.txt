

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

# if(COMMAND cmake_policy)
# 	cmake_policy(SET CMP0003 OLD)
# 	cmake_policy(SET CMP0015 OLD)
# endif(COMMAND cmake_policy)

# link_directories(${ZED_LIBRARY_DIR})
# link_directories(${CUDA_LIBRARY_DIRS})

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

ADD_LIBRARY(libzed zed.cpp ../include/zed.hpp)
TARGET_LINK_LIBRARIES(libzed PUBLIC ${ZED_LIBS} ${OpenCV_LIBS})
set_target_properties(libzed PROPERTIES
    CXX_STANDARD                14
    CXX_EXTENSIONS              OFF
    CXX_STANDARD_REQUIRED       ON
    LIBRARY_OUTPUT_DIRECTORY    "${CMAKE_HOME_DIRECTORY}/lib"
)
target_include_directories(libzed 
  PUBLIC 
    "${CMAKE_HOME_DIRECTORY}/include"
    ${CUDA_INCLUDE_DIRS}
    
)
