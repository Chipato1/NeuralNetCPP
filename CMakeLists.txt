# cmake version to be used
cmake_minimum_required( VERSION 3.0 )

# project name
project( MultiPerceptron )

# flags

# files
add_subdirectory( ./data )

# include
include_directories( ./include ./src ./libs/Logger/include usr/include/eigen3)

# target
add_executable( binary ./main.cpp ./source/Affine_Layer.cpp ./source/ReLu_Layer.cpp ./source/Network.cpp ./source/Trainer.cpp ./source/Dataloader.cpp  ./source/Sigm_Layer.cpp ./source/Trainer_MD.cpp ./source/Sofm_Layer.cpp)