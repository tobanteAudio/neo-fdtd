#!/bin/sh

set -e

build_dir=build
# build_dir=cmake-build-sycl

root_dir="$(cd "$(dirname "$0")" && pwd)"
python_dir="$root_dir/src/python"
engine_exe="$root_dir/$build_dir/src/cpp/main_2d/pffdtd_2d"

sim_name="Modes2D"
sim_dir="$root_dir/data/sim_data/$sim_name/cpu"
sim_setup="${sim_name}.py"
model_dir="$root_dir/data/models/$sim_name"

# Delete old sim
rm -rf "$sim_dir"

# Generate model
cd "$model_dir"
python "$sim_setup"

# Run sim
DPCPP_CPU_PLACES=cores DPCPP_CPU_CU_AFFINITY=spread DPCPP_CPU_NUM_CUS=16 "$engine_exe" -s "$sim_dir"
# pffdtd sim2d run --sim_dir "$sim_dir"

# Report
python -m pffdtd.sim2d.report --sim_dir="$sim_dir" "$sim_dir/out.h5"