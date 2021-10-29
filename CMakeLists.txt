# cmake version to be used
cmake_minimum_required( VERSION 3.0 )

# project name
project( MultiPerceptron )

# flags

# files

# include
include_directories( ./include ./src ./libs/Logger/include ./libs/Eigen ./libs/Eigen/src)

# target
add_executable( binary ./main.cpp ./source/Layer.cpp)
# external libs - 3rd party libs
# add_subdirectory( ./libs/Logger )
# target_link_libraries( binary logger )

# add_subdirectory( ./libs/Randomize )
# target_link_libraries( binary randomize )

#add_library(randomize STATIC IMPORTED)
#set_property(TARGET randomize PROPERTY IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/Randomize/librandomize.a)
#ftarget_link_libraries( binary randomize )