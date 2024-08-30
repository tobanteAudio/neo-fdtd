# Build

## SYCL

```sh
# Build with sycl compiler & custom conan profile
cmake -S. -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -D PFFDTD_ENABLE_INTEL_SYCL=ON -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=external/cmake-conan/conan_provider.cmake -DCONAN_HOST_PROFILE=/full/path/to/pffdtd/cmake/profile/linux_sycl
cmake --build build

# Using hyper-threads is usally a slow down. Use the number of physical cores.
DPCPP_CPU_PLACES=cores DPCPP_CPU_CU_AFFINITY=spread DPCPP_CPU_NUM_CUS=16 ./build/src/main_2d/pffdtd_2d ./data/sim_data/Diffusor/cpu/sim.h5
```
