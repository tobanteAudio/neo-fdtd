cmake_minimum_required(VERSION 3.24)
project(pffdtd VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(pffdtd)
target_compile_definitions(pffdtd PRIVATE PRECISION=1 USING_CUDA=0)
target_sources(pffdtd 
    PRIVATE
        src/native/cpu_engine.c
        src/native/cpu_engine.h
        src/native/fdtd_common.c
        src/native/fdtd_common.h
        src/native/fdtd_data.c
        src/native/fdtd_data.h
        src/native/helper_funcs.c
        src/native/helper_funcs.h
        src/native/main.c
)

find_package(HDF5 REQUIRED)
target_link_libraries(pffdtd PRIVATE HDF5::HDF5)

find_package(OpenMP REQUIRED)
target_link_libraries(pffdtd PRIVATE OpenMP::OpenMP_C)

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
    target_compile_options(pffdtd PRIVATE /W4)
else ()
    target_compile_options(pffdtd PRIVATE -Wno-deprecated-declarations -Wno-format)
endif ()