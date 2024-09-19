# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

project(pffdtd_2d)

add_executable(pffdtd_2d main.cpp)
target_link_libraries(pffdtd_2d PRIVATE CLI11::CLI11 pffdtd::pffdtd)
