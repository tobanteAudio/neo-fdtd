# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

project(pffdtd_3d)

add_executable(pffdtd_3d main.cpp)
target_link_libraries(pffdtd_3d PRIVATE pffdtd::pffdtd)
