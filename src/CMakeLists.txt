

add_library(superpoint SHARED
test.cpp
../include/test.hpp
)
target_compile_features(superpoint PUBLIC cxx_std_14)
target_link_libraries(superpoint PUBLIC
${TORCH_LIBRARIES}
${OpenCV_LIBS}
)
set_target_properties(superpoint PROPERTIES
CXX_EXTENSIONS  OFF
CXX_STANDARD_REQUIRED   ON
LIBRARY_OUTPUT_DIRECTORY    "${CMAKE_HOME_DIRECTORY}/lib"
)
target_include_directories(superpoint PUBLIC "../include")