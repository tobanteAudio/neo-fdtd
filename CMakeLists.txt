cmake_minimum_required(VERSION 3.24)
project(pffdtd VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PFFDTD_ENABLE_SYCL "Build with sycl" OFF)

find_package(HDF5 REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(src/cpp)
add_subdirectory(src/3D)

if(PFFDTD_ENABLE_SYCL)
    find_package(IntelSYCL REQUIRED)
    add_subdirectory(src/2D)
endif()
