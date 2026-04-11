cmake_minimum_required(VERSION 3.10)
project(MLPilot)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(mlpilot cpp_cli/pilot.cpp)