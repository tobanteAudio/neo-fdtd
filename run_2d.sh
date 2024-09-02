#!/bin/sh

set -e

build_dir=build
# build_dir=cmake-build-sycl

root_dir="$(cd "$(dirname "$0")" && pwd)"
python_dir="$root_dir/src/python"
engine_exe="$root_dir/$build_dir/src/cpp/main_2d/pffdtd_2d"

sim_name="Diffusor"
sim_dir="$root_dir/data/sim_data/$sim_name/cpu"

fmax=4000
duration=0.050

# Delete old sim
rm -rf "$sim_dir"

# Generate model
python -m pffdtd.sim2d.fdtd --verbose --save --data_dir="$sim_dir" --duration="$duration" --fmax="$fmax"

# Run sim
DPCPP_CPU_PLACES=cores DPCPP_CPU_CU_AFFINITY=spread DPCPP_CPU_NUM_CUS=16 "$engine_exe" -s "$sim_dir"

# Report
python -m pffdtd.sim2d.report --sim_dir="$sim_dir" "$sim_dir/out.h5"