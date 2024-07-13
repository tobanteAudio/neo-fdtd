project(pffdtd_3d)

add_executable(pffdtd_3d)
target_compile_definitions(pffdtd_3d PRIVATE USING_CUDA=0)
target_sources(pffdtd_3d
    PRIVATE
        engine_cpu.cpp
        engine_cpu.hpp
        main.cpp
)


target_link_libraries(pffdtd_3d PRIVATE pffdtd::pffdtd OpenMP::OpenMP_CXX)

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
    target_compile_options(pffdtd_3d PRIVATE /W4)
else ()
    target_compile_options(pffdtd_3d PRIVATE -Wno-deprecated-declarations)
endif ()
