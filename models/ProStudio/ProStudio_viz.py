# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

from pffdtd.sim3d.setup import sim_setup_3d

sim_setup_3d(
    model_json_file='model.json',
    mat_folder='../../materials',
    source_num=2,
    insig_type='dhann30',  # for viz
    diff_source=False,
    mat_files_dict={
        'ATC Left': 'floor_wood.h5',
        'ATC Right': 'floor_wood.h5',
        'Ceiling': 'absorber_8000_200mm_gap_200mm.h5',
        'Console': 'door_iron.h5',
        'Couch': 'leather_arm_chair.h5',
        # 'Diffusor': 'floor_wood.h5',
        'Floor': 'floor_wood_on_concrete.h5',
        'Outboard': 'door_iron.h5',
        'Rack': 'floor_wood.h5',
        'Raised Floor': 'floor_wood.h5',
        'Walls Back': 'absorber_8000_200mm_gap_200mm.h5',
        'Walls Front': 'absorber_8000_200mm_gap_100mm.h5',
        'Walls Side': 'absorber_8000_50mm.h5',
        'Windows': 'glas_thick.h5',
    },
    duration=0.1,
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=7.7,
    fmax=800.0,
    save_folder='../../sim_data/ProStudio/viz',
    compress=0,
    draw_vox=True,
    draw_backend='polyscope',
)

# then run with python and 3D visualization:
#   pffdtd sim3d engine --sim_dir='../../sim_data/ProStudio/viz' --plot --draw_backend='mayavi' --json_model='model.json'
