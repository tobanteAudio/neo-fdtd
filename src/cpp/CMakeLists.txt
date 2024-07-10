project(pffdtd)

add_library(pffdtd INTERFACE)
add_library(pffdtd::pffdtd ALIAS pffdtd)
target_include_directories(pffdtd INTERFACE ${PROJECT_SOURCE_DIR})
target_link_libraries(pffdtd INTERFACE HDF5::HDF5)
