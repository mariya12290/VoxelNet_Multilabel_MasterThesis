cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(Object_Detection)

ADD_LIBRARY(DetectionModule 
            src/load_graphs.cpp 
            src/pre_processing.cpp)

#tensorflow library
find_package(TensorflowCC REQUIRED)

#Eigen library 
find_package(Eigen3 3.3.90 EXACT REQUIRED COMPONENTS )
include_directories(${EIGEN3_INCLUDE_DIR})
link_directories(${EIGEN3_LIBRARIES_DIRS})
add_definitions(${EIGEN3_DEFINITIONS})

#pcl library
find_package(PCL 1.11.1.99 EXACT REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIR})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARIES_DIRS})

add_definitions(${PCL_DEFINITIONS})

#Boost library
find_package(Boost 1.55.0 REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

target_include_directories(DetectionModule INTERFACE ${EIGEN3_INCLUDE_DIR})

target_compile_definitions(DetectionModule PUBLIC -D_GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(DetectionModule -lstdc++fs TensorflowCC::TensorflowCC  ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${Boost_LIBRARIES})

#https://stackoverflow.com/questions/39598323/how-to-properly-link-libraries-with-cmake

add_executable(object_detection src/main.cpp)


target_link_libraries(object_detection  PUBLIC -pthread -lprotobuf -lpthread DetectionModule)
