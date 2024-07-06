cmake_minimum_required(VERSION 3.24)
project(pffdtd VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(pffdtd)
target_compile_definitions(pffdtd PRIVATE PRECISION=2 USING_CUDA=0)
target_sources(pffdtd
    PRIVATE
        src/native/cpu_engine.cpp
        src/native/cpu_engine.hpp
        src/native/fdtd_common.cpp
        src/native/fdtd_common.hpp
        src/native/fdtd_data.cpp
        src/native/fdtd_data.hpp
        src/native/helper_funcs.cpp
        src/native/helper_funcs.hpp
        src/native/main.cpp
)

find_package(HDF5 REQUIRED)
target_link_libraries(pffdtd PRIVATE HDF5::HDF5)

find_package(OpenMP REQUIRED)
target_link_libraries(pffdtd PRIVATE OpenMP::OpenMP_CXX)

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
    target_compile_options(pffdtd PRIVATE /W4)
else ()
    target_compile_options(pffdtd PRIVATE -Wno-deprecated-declarations)
endif ()
