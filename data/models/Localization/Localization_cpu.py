# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

from pffdtd.sim3d.setup import sim_setup_3d


sim_setup_3d(
    model_json_file='model.json',
    mat_folder='../../materials',
    source_num=1,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={
        'Ceiling': 'sabine_9512.h5',
        'Floor': 'sabine_9512.h5',
        'Walls': 'sabine_9512.h5',
    },
    duration=0.4,
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=10.5,
    fmax=800.0,
    save_folder='../../sim_data/Localization/cpu',
    save_folder_gpu='../../sim_data/Localization/gpu',
    compress=0,
)
