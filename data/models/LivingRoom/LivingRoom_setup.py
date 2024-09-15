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
        'Book Shelf': 'floor_wood.h5',
        'Ceiling': 'concrete_painted.h5',
        'Coffee Table': 'floor_wood.h5',
        'Couch': 'absorber_8000_100mm.h5',
        'Desk': 'floor_wood.h5',
        'Door': 'floor_wood.h5',
        'Floor': 'floor_wood.h5',
        'Kallax': 'floor_wood.h5',
        'Monitors': 'floor_wood.h5',
        'Speakers': 'floor_wood.h5',
        'Speaker Stands': 'door_iron.h5',
        'Table': 'floor_wood.h5',
        'TV 42': 'floor_wood.h5',
        'TV 55': 'floor_wood.h5',
        'TV Table': 'floor_wood.h5',
        'Walls': 'concrete_painted.h5',
        'Window': 'glas_thick.h5',
    },
    duration=2.0,
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=10.5,
    fmax=800.0,
    save_folder='../../sim_data/LivingRoom/cpu',
    save_folder_gpu='../../sim_data/LivingRoom/gpu',
    compress=0,
    draw_vox=True,
    draw_backend='polyscope',
)
