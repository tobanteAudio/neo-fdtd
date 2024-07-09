.PHONY: all

CXX=/opt/intel/oneapi/compiler/latest/bin/icpx
LIBS=-lhdf5

# env:
# 	source /opt/intel/oneapi/setvars.sh

all:
	${CXX} -fsycl -O3 -D NDEBUG ${LIBS} fdtd.cpp
