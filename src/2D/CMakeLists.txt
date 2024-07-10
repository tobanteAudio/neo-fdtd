project(pffdtd_2d)

add_executable(pffdtd_2d)
target_sources(pffdtd_2d PRIVATE fdtd.cpp)
add_sycl_to_target(TARGET pffdtd_2d SOURCES fdtd.cpp)
target_link_libraries(pffdtd_2d PRIVATE HDF5::HDF5)
