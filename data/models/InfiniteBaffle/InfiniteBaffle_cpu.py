# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

from pffdtd.sim3d.setup import sim_setup_3d

sim_setup_3d(
    model_json_file='model.json',
    mat_folder='../../materials',
    source_num=1,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={},
    duration=0.2,
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=7.7,
    fmax=16000.0,
    save_folder='../../sim_data/InfiniteBaffle/cpu',
    save_folder_gpu='../../sim_data/InfiniteBaffle/gpu',
    draw_vox=True,
    draw_backend='polyscope',
    compress=0,
    rot_az_el=[0, 0],
    bmax=[3.0, 2.0, 3.0],
    bmin=[0, 0, 0],
)
