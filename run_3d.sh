#!/bin/sh

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

set -e

root_dir="$(cd "$(dirname "$0")" && pwd)"
pffdtd_engine="$root_dir/build/src/cpp/pffdtd-engine"
pffdtd_engine="$root_dir/cmake-build-sycl/src/cpp/pffdtd-engine"

sim_name="ProStudio"
sim_setup="${sim_name}.py"
sim_dir="$root_dir/sim_data/$sim_name/gpu"

model_dir="$root_dir/models/$sim_name"
materials_dir="$root_dir/materials"

fmin=20
fmax=2000
smoothing=0

# Delete old sim
rm -rf "$sim_dir"

# Generate materials, model & sim data
cd "$model_dir"
pffdtd sim3d setup "$sim_setup"

# Run sim
$pffdtd_engine sim3d -s "$sim_dir"
# pffdtd sim3d engine --sim_dir="$sim_dir" --plot --draw_backend="mayavi" --json_model="${model_dir}/model.json"

# Post-process
pffdtd sim3d process-outputs --sim_dir="$sim_dir" --fcut_lowpass "$fmax" --order_lowpass=8 --symmetric_lowpass --fcut_lowcut "$fmin" --order_lowcut=4 --air_abs_filter="none" --save_wav --plot
pffdtd analysis response --fmin=10 --target="-2.5" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R002_out_normalised.wav
pffdtd analysis response --fmin=10 --target="-2.0" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R003_out_normalised.wav
pffdtd analysis response --fmin=10 --target="-1.5" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R004_out_normalised.wav
pffdtd analysis response --fmin=10 --target="-1.0" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R005_out_normalised.wav
pffdtd analysis response --fmin=10 --target="-0.5" --smoothing=$smoothing --fmax=$fmax $sim_dir/R001_out_normalised.wav $sim_dir/R006_out_normalised.wav
pffdtd analysis waterfall $sim_dir/R001_out_normalised.wav
pffdtd analysis t60 --fmin=$fmin --fmax="$fmax" --target=0.3 $sim_dir/R001_out_normalised.wav
pffdtd analysis t60 --sim_dir="$sim_dir" --fmin=$fmin --fmax="$fmax" --target=0.25
pffdtd analysis room-modes --sim_dir="$sim_dir" --fmin=$fmin --fmax=$fmax --num_modes=20 --width=3.65 --length=6.0 --height=3.12
