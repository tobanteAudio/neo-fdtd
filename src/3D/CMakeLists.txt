project(pffdtd_3d)

add_executable(pffdtd_3d)
target_compile_definitions(pffdtd_3d PRIVATE PRECISION=2 USING_CUDA=0)
target_sources(pffdtd_3d
    PRIVATE
        cpu_engine.cpp
        cpu_engine.hpp
        fdtd_common.cpp
        fdtd_common.hpp
        fdtd_data.cpp
        fdtd_data.hpp
        helper_funcs.cpp
        helper_funcs.hpp
        main.cpp
)


target_link_libraries(pffdtd_3d PRIVATE HDF5::HDF5)
target_link_libraries(pffdtd_3d PRIVATE OpenMP::OpenMP_CXX)

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") OR (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
    target_compile_options(pffdtd_3d PRIVATE /W4)
else ()
    target_compile_options(pffdtd_3d PRIVATE -Wno-deprecated-declarations)
endif ()
