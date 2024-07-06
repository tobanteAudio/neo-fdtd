#!/bin/sh

set -e
root_dir="$(cd "$(dirname "$0")" && pwd)"

python_dir="$root_dir/src/python"
engine_exe="$root_dir/build/pffdtd"

sim_dir="$root_dir/data/sim_data/Tobi/cpu"
sim_name="test_script_Tobi_cart_cpu.py"
model_dir="$root_dir/data/models/Tobi_Office"

# Delete old sim
rm -rf "$sim_dir"

# Create model
cd "$model_dir"
python generate_model.py

# Create sim data
cd "$python_dir"
python build_mats.py
python "$sim_name"

# Run sim
cd "$sim_dir"
OMP_NUM_THREADS=16 $engine_exe

# Post-process
cd "$python_dir"
python -m fdtd.process_outputs --data_dir="$sim_dir" --fcut_lowpass 1000.0 --N_order_lowpass=8 --symmetric --fcut_lowcut 10.0 --N_order_lowcut=4 --air_abs_filter="stokes" --save_wav
python calc_t60.py "$sim_dir"
python -m fdtd.process_outputs --data_dir="$sim_dir" --fcut_lowpass 1000.0 --N_order_lowpass=8 --symmetric --fcut_lowcut 10.0 --N_order_lowcut=4 --air_abs_filter="stokes" --plot
