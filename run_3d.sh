#!/bin/sh

set -e

root_dir="$(cd "$(dirname "$0")" && pwd)"
python_dir="$root_dir/src/python"
engine_exe="$root_dir/cmake-build-sycl/src/cpp/main_3d/pffdtd_3d"

sim_name="Modes"
sim_setup="${sim_name}_cpu.py"
sim_model_gen="${sim_name}_model.py"
sim_dir="$root_dir/data/sim_data/$sim_name/cpu"

model_dir="$root_dir/data/models/$sim_name"
materials_dir="$root_dir/data/materials"

fmax=1000

# Delete old sim
rm -rf "$sim_dir"

# Generate model
cd "$python_dir"
python "$sim_model_gen"

# Generate sim data
cd "$python_dir"
python -m materials.build "$materials_dir"
python "$sim_setup"

# Run sim
cd "$sim_dir"
$engine_exe

# Post-process
cd "$python_dir"
python -m sim3d.process_outputs --data_dir="$sim_dir" --fcut_lowpass "$fmax" --N_order_lowpass=8 --symmetric --fcut_lowcut 20.0 --N_order_lowcut=4 --air_abs_filter="none" --save_wav --plot
python -m analysis.t60 --data_dir="$sim_dir" --fmin=20 --fmax="$fmax"
python -m analysis.room_modes --data_dir="$sim_dir" --fmin=10 --fmax=200 --modes=10
