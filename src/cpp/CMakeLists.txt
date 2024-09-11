# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

project(pffdtd)

add_library(pffdtd)
add_library(pffdtd::pffdtd ALIAS pffdtd)
target_include_directories(pffdtd PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_definitions(pffdtd PUBLIC PRECISION=2)

target_link_libraries(pffdtd
    PUBLIC
        fmt::fmt
        HDF5::HDF5
        OpenMP::OpenMP_CXX
        std::mdspan
)

target_sources(pffdtd
    PRIVATE
        pffdtd/config.hpp
        pffdtd/exception.hpp
        pffdtd/hdf.hpp
        pffdtd/mdspan.hpp
        pffdtd/progress.cpp
        pffdtd/progress.hpp
        pffdtd/simulation_2d.cpp
        pffdtd/simulation_2d.hpp
        pffdtd/simulation_3d.cpp
        pffdtd/simulation_3d.hpp
        pffdtd/sycl.hpp
        pffdtd/utility.cpp
        pffdtd/utility.hpp
)

target_compile_definitions(pffdtd PUBLIC _CRT_SECURE_NO_WARNINGS=1)

if(PFFDTD_ENABLE_ACPP_SYCL OR PFFDTD_ENABLE_INTEL_SYCL)
    target_compile_definitions(pffdtd PUBLIC PFFDTD_HAS_SYCL=1)
endif()

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
    target_compile_options(pffdtd PRIVATE /W3)
else ()
    target_compile_options(pffdtd PRIVATE -Wno-deprecated-declarations)
endif ()
