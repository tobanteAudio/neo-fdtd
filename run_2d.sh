#!/bin/sh

set -e

build_dir=cmake-build-sycl
root_dir="$(cd "$(dirname "$0")" && pwd)"
python_dir="$root_dir/src/python"
engine_exe="$root_dir/$build_dir/src/cpp/main_2d/pffdtd_2d"

sim_name="Diffusor"
sim_dir="$root_dir/data/sim_data/$sim_name/cpu"

fmax=1000
duration=0.045

# Delete old sim
rm -rf "$sim_dir"

# Generate model
cd "$python_dir"
python -m sim2d.fdtd --save --data_dir="$sim_dir" --duration="$duration" --fmax="$fmax"

# Run sim
DPCPP_CPU_PLACES=cores DPCPP_CPU_CU_AFFINITY=spread DPCPP_CPU_NUM_CUS=16 "$engine_exe" -s "$sim_dir/sim.h5"
