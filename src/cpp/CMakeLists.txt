# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

project(pffdtd)

add_library(pffdtd)
add_library(pffdtd::pffdtd ALIAS pffdtd)
target_include_directories(pffdtd PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_definitions(pffdtd PUBLIC _CRT_SECURE_NO_WARNINGS=1)

target_link_libraries(pffdtd
    PUBLIC
        fmt::fmt
        HDF5::HDF5
        OpenMP::OpenMP_CXX
        std::mdspan
)

target_sources(pffdtd
    PRIVATE
        pffdtd/engine_2d_cpu.cpp
        pffdtd/engine_2d_cpu.hpp
        pffdtd/engine_3d_cpu.cpp
        pffdtd/engine_3d_cpu.hpp
        pffdtd/exception.hpp
        pffdtd/hdf.cpp
        pffdtd/hdf.hpp
        pffdtd/mdspan.hpp
        pffdtd/progress.cpp
        pffdtd/progress.hpp
        pffdtd/simulation_2d.cpp
        pffdtd/simulation_2d.hpp
        pffdtd/simulation_3d.cpp
        pffdtd/simulation_3d.hpp
        pffdtd/time.hpp
        pffdtd/utility.hpp
)

if(PFFDTD_ENABLE_CUDA)
    target_sources(pffdtd PRIVATE pffdtd/engine_3d_cuda.cu pffdtd/engine_3d_cuda.hpp)
    target_compile_definitions(pffdtd PUBLIC PFFDTD_HAS_CUDA=1)
endif()

if(PFFDTD_HAS_SYCL)
    target_sources(pffdtd PRIVATE pffdtd/engine_2d_sycl.cpp pffdtd/engine_2d_sycl.hpp)
    target_sources(pffdtd PRIVATE pffdtd/engine_3d_sycl.cpp pffdtd/engine_3d_sycl.hpp)
    target_sources(pffdtd PRIVATE pffdtd/sycl.cpp pffdtd/sycl.hpp)
    add_sycl_to_target(TARGET pffdtd SOURCES pffdtd/engine_2d_sycl.cpp pffdtd/engine_3d_sycl.cpp pffdtd/sycl.cpp)
    target_compile_definitions(pffdtd PUBLIC PFFDTD_HAS_SYCL=1)

    if(PFFDTD_ENABLE_SYCL_ONEAPI)
        set_source_files_properties(pffdtd/engine_2d_sycl.cpp PROPERTIES COMPILE_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda")
        set_source_files_properties(pffdtd/engine_3d_sycl.cpp PROPERTIES COMPILE_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda")
        target_link_libraries(pffdtd PRIVATE "-fsycl-targets=nvptx64-nvidia-cuda")
    endif()
endif()

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
    target_compile_options(pffdtd PUBLIC /W3)
else ()
    target_compile_options(pffdtd PUBLIC -Wall -Wextra -Wno-deprecated-declarations)
endif ()

add_executable(pffdtd-engine main.cpp)
target_link_libraries(pffdtd-engine PRIVATE CLI11::CLI11 pffdtd::pffdtd)
