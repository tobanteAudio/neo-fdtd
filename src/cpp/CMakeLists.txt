project(pffdtd)

add_library(pffdtd)
add_library(pffdtd::pffdtd ALIAS pffdtd)
target_include_directories(pffdtd PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_definitions(pffdtd PUBLIC PRECISION=2)

target_link_libraries(pffdtd
    PUBLIC
        fmt::fmt
        HDF5::HDF5
        opencv_videoio
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
        pffdtd/video.cpp
        pffdtd/video.hpp
)
