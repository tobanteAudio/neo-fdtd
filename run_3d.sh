#!/bin/sh

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

set -e

root_dir="$(cd "$(dirname "$0")" && pwd)"
engine_exe="$root_dir/build/src/cpp/main_3d/pffdtd_3d"
engine_exe="$root_dir/cmake-build-cuda/src/cpp/main_3d/pffdtd_3d"

sim_name="LivingRoom"
sim_setup="${sim_name}_setup.py"
sim_model_gen="${sim_name}_model.py"
sim_dir="$root_dir/data/sim_data/$sim_name/gpu"

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
pffdtd materials build "$materials_dir"
python "$sim_setup"

# Run sim
$engine_exe "$sim_dir"

# Post-process
pffdtd sim3d process-outputs --sim_dir="$sim_dir" --fcut_lowpass "$fmax" --order_lowpass=8 --symmetric_lowpass --fcut_lowcut "$fmin" --order_lowcut=4 --air_abs_filter="stokes" --save_wav --plot
pffdtd analysis response --fmin=10 --target="-2.5" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R002_out_normalised.wav
# pffdtd analysis response --fmin=10 --target="-2.0" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R003_out_normalised.wav
# pffdtd analysis response --fmin=10 --target="-1.5" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R004_out_normalised.wav
# pffdtd analysis response --fmin=10 --target="-1.0" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R005_out_normalised.wav
# pffdtd analysis response --fmin=10 --target="-0.5" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R006_out_normalised.wav
pffdtd analysis waterfall $sim_dir/R001_out_normalised.wav
pffdtd analysis t60 --fmin=$fmin --fmax="$fmax" --target=0.3 $sim_dir/R001_out_normalised.wav
pffdtd analysis t60 --sim_dir="$sim_dir" --fmin=$fmin --fmax="$fmax" --target=0.25
pffdtd analysis room-modes --sim_dir="$sim_dir" --fmin=$fmin --fmax=$fmax --num_modes=20 --width=3.65 --length=6.0 --height=3.12
