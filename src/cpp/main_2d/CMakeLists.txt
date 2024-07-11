project(pffdtd_2d)

add_executable(pffdtd_2d)
target_link_libraries(pffdtd_2d PRIVATE CLI11::CLI11 pffdtd::pffdtd)
target_compile_options(pffdtd_2d PRIVATE -Wno-deprecated-declarations)
target_sources(pffdtd_2d PRIVATE main.cpp engine.cpp)
add_sycl_to_target(TARGET pffdtd_2d SOURCES engine.cpp)
