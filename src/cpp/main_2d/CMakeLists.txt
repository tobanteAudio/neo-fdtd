project(pffdtd_2d)

add_executable(pffdtd_2d main.cpp engine_native.cpp)
target_link_libraries(pffdtd_2d PRIVATE CLI11::CLI11 pffdtd::pffdtd)
target_compile_options(pffdtd_2d PRIVATE -Wall -Wextra -Wpedantic)

if(PFFDTD_ENABLE_ACPP_SYCL OR PFFDTD_ENABLE_INTEL_SYCL)
    target_sources(pffdtd_2d PRIVATE engine_sycl.cpp)
    add_sycl_to_target(TARGET pffdtd_2d SOURCES engine_sycl.cpp)
endif()
