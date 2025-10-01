cmake_minimum_required(VERSION 3.10)
project(TrajectoryFitting)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(TrajectoryFitting main.cpp)

target_link_libraries(TrajectoryFitting Ceres::ceres)
target_link_libraries(TrajectoryFitting ${OpenCV_LIBS})
