cmake_minimum_required(VERSION 3.28)
project(HW3_2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/3rdParty/opencl/include)

add_executable(HW3_2 main.cpp)

# linking the OpenCL library
if (APPLE)
    find_package(OpenCL REQUIRED)
    target_link_libraries(HW3_2 OpenCL::OpenCL)
else ()
    link_directories(${CMAKE_SOURCE_DIR}/3rdParty/opencl/lib)
    target_link_libraries(HW3_2 OpenCL)
endif ()
