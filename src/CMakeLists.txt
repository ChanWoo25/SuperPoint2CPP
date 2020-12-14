

### LIBRARY - Super
add_library(Super SHARED
SuperPoint.cpp
SPDetector.cpp
#SPExtractor.cpp
Tools.cpp
)
target_compile_features(Super PUBLIC cxx_std_14)
target_link_libraries(Super PUBLIC
${TORCH_LIBRARIES}
${OpenCV_LIBS}
)
set_target_properties(Super PROPERTIES
LIBRARY_OUTPUT_DIRECTORY    "${CMAKE_HOME_DIRECTORY}/lib"
)
target_include_directories(Super PUBLIC "../include")