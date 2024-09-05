# Build

```shell
cd /path/to/pffdtd
git clone https://github.com/conan-io/cmake-conan.git -b develop2 external/cmake-conan
```

## Linux

### Intel oneAPI SYCL

```shell
# Build with sycl compiler & custom conan profile
cmake -S. -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -D PFFDTD_ENABLE_INTEL_SYCL=ON -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=external/cmake-conan/conan_provider.cmake -DCONAN_HOST_PROFILE=/full/path/to/pffdtd/cmake/profile/linux_sycl
cmake --build build

# Using hyper-threads is usally a slow down. Use the number of physical cores.
DPCPP_CPU_PLACES=cores DPCPP_CPU_CU_AFFINITY=spread DPCPP_CPU_NUM_CUS=16 ./build/src/main_2d/pffdtd_2d ./data/sim_data/Diffusor/cpu/sim.h5
```

## Windows

- Visual Studio
- Intel oneAPI DPCPP

```shell
winget install -e --id Kitware.CMake
winget install -e --id Ninja-build.Ninja
winget install -e --id JFrog.Conan
```

### MSVC

```shell
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_PROJECT_TOP_LEVEL_INCLUDES=external/cmake-conan/conan_provider.cmake
cmake --build build
```

### Intel oneAPI SYCL

```shell
# Use "Intel oneAPI command prompt" application
# Or: Set environment variables
. 'C:\Program Files (x86)\Intel\oneAPI\setvars.bat'
```

```shell
cmake -S . -B build -G Ninja -D PFFDTD_ENABLE_INTEL_SYCL=ON -D CMAKE_BUILD_TYPE=Release -D CMAKE_C_COMPILER=icx -D CMAKE_CXX_COMPILER=icx -D CMAKE_PROJECT_TOP_LEVEL_INCLUDES=external/cmake-conan/conan_provider.cmake -D CONAN_HOST_PROFILE=../cmake/profile/windows_sycl
cmake --build build
```
