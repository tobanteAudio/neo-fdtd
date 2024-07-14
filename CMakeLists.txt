cmake_minimum_required(VERSION 3.24)
project(pffdtd VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PFFDTD_ENABLE_ACPP_SYCL "Build with AdaptiveCpp SYCL" OFF)
option(PFFDTD_ENABLE_INTEL_SYCL "Build with Intel SYCL" OFF)

include(FetchContent)
FetchContent_Declare(mdspan GIT_REPOSITORY "https://github.com/kokkos/mdspan" GIT_TAG "stable" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(mdspan)

find_package(CLI11 REQUIRED)
find_package(fmt REQUIRED)
find_package(HDF5 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core videoio)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)

if(PFFDTD_ENABLE_ACPP_SYCL)
    find_package(AdaptiveCpp REQUIRED)
endif()

if(PFFDTD_ENABLE_INTEL_SYCL)
    find_package(IntelSYCL REQUIRED)
endif()

add_subdirectory(src/cpp)
add_subdirectory(src/cpp/main_3d)

if(PFFDTD_ENABLE_ACPP_SYCL OR PFFDTD_ENABLE_INTEL_SYCL)
    add_subdirectory(src/cpp/main_2d)
endif()
