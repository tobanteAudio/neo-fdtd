project(pffdtd_3d)

add_executable(pffdtd_3d)
target_compile_definitions(pffdtd_3d PRIVATE USING_CUDA=0)
target_sources(pffdtd_3d
    PRIVATE
        engine_cpu.cpp
        engine_cpu.hpp
        main.cpp
)

target_link_libraries(pffdtd_3d PRIVATE pffdtd::pffdtd)
