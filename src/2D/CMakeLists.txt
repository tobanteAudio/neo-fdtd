project(pffdtd_2d)

find_package(IntelSYCL REQUIRED)

add_executable(pffdtd_2d)
target_sources(pffdtd_2d PRIVATE fdtd.cpp)
add_sycl_to_target(TARGET pffdtd_2d SOURCES fdtd.cpp)
target_compile_options(pffdtd_2d PRIVATE -Wno-deprecated-declarations)
target_link_libraries(pffdtd_2d PRIVATE HDF5::HDF5)
