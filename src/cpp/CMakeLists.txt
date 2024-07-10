project(pffdtd)

add_library(pffdtd)
add_library(pffdtd::pffdtd ALIAS pffdtd)
target_include_directories(pffdtd PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(pffdtd PUBLIC fmt::fmt HDF5::HDF5)
target_sources(pffdtd
    PRIVATE
        pffdtd/hdf.hpp
        pffdtd/simulation_2d.cpp
        pffdtd/simulation_2d.hpp
        pffdtd/sycl.hpp
)
