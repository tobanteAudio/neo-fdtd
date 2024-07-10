# Main 2D

## Raw SYCL

```makefile
.PHONY: all

CXX=/opt/intel/oneapi/compiler/latest/bin/icpx
FLAGS=-fsycl -std=c++20 -D NDEBUG -O3 -march=native
LIBS=-lhdf5
WARNINGS=-Wall -Wno-deprecated-declarations

# env:
# 	source /opt/intel/oneapi/setvars.sh

all:
	${CXX} ${FLAGS} ${LIBS} ${WARNINGS} fdtd.cpp
```

```sh
SETVARS_ARGS="--force" cmake --build build
```
