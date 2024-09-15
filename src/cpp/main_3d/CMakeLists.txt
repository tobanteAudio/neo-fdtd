# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

project(pffdtd_3d)

add_executable(pffdtd_3d)
target_link_libraries(pffdtd_3d PRIVATE pffdtd::pffdtd)
target_sources(pffdtd_3d
    PRIVATE
        engine_cpu.cpp
        engine_cpu.hpp
        main.cpp
)

if(PFFDTD_ENABLE_CUDA)
    target_sources(pffdtd_3d PRIVATE engine_gpu.cu engine_gpu.hpp)
    target_compile_definitions(pffdtd_3d PRIVATE PFFDTD_HAS_CUDA=1)
else()
    target_compile_definitions(pffdtd_3d PRIVATE PFFDTD_HAS_CUDA=0)
endif()
