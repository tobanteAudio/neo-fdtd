#!/bin/sh

set -e

root_dir="$(cd "$(dirname "$0")" && pwd)"
engine_exe="$root_dir/build/src/cpp/main_3d/pffdtd_3d"

sim_name="ProStudio"
sim_setup="${sim_name}_cpu.py"
sim_model_gen="${sim_name}_model.py"
sim_dir="$root_dir/data/sim_data/$sim_name/cpu"

model_dir="$root_dir/data/models/$sim_name"
materials_dir="$root_dir/data/materials"

fmin=25
fmax=800
smoothing=0

# Delete old sim
rm -rf "$sim_dir"

# Generate model
cd "$model_dir"
python "$sim_model_gen"

# Generate sim data
python -m pffdtd.materials.build "$materials_dir"
python "$sim_setup"

# Run sim
cd "$sim_dir"
$engine_exe

# Post-process
python -m pffdtd.sim3d.process_outputs --data_dir="$sim_dir" --fcut_lowpass "$fmax" --N_order_lowpass=8 --symmetric --fcut_lowcut "$fmin" --N_order_lowcut=4 --air_abs_filter="stokes" --save_wav --plot
python -m pffdtd.analysis.response --fmin=10 --target="-2.5" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R002_out_normalised.wav
python -m pffdtd.analysis.response --fmin=10 --target="-2.0" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R003_out_normalised.wav
python -m pffdtd.analysis.response --fmin=10 --target="-1.5" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R004_out_normalised.wav
python -m pffdtd.analysis.response --fmin=10 --target="-1.0" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R005_out_normalised.wav
python -m pffdtd.analysis.response --fmin=10 --target="-0.5" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R006_out_normalised.wav
python -m pffdtd.analysis.waterfall $sim_dir/R001_out_normalised.wav
python -m pffdtd.analysis.t60 --fmin=$fmin --fmax="$fmax" --target=0.3 $sim_dir/R001_out_normalised.wav
python -m pffdtd.analysis.t60 --data_dir="$sim_dir" --fmin=$fmin --fmax="$fmax" --target=0.25
# python -m pffdtd.analysis.room_modes --data_dir="$sim_dir" --fmin=$fmin --fmax=$fmax --modes=20
