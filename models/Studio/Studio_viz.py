# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

from pffdtd.sim3d.setup import sim_setup_3d

sim_setup_3d(
    model_json_file='model.json',
    mat_folder='../../materials',
    source_num=1,
    insig_type='dhann30',  # for viz
    diff_source=False,
    mat_files_dict={
        'Absorber M': 'absorber_8000_100mm.h5',
        'Absorber L': 'absorber_8000_200mm.h5',
        'Ceiling': 'concrete_painted.h5',
        'Diffusor': 'door_wood.h5',
        'Floor': 'floor_wood.h5',
        'Sofa': 'absorber_8000_100mm.h5',
        'Speaker_Cabinet': 'door_wood.h5',
        'Table': 'door_wood.h5',
        'Walls': 'concrete_painted.h5',
    },
    duration=0.1,  # duration in seconds
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=7.7,  # for 2% phase velocity error at fmax
    fmax=500.0,
    save_folder='../../sim_data/Studio/viz',  # can run python from here
    compress=0,
    draw_vox=True,
    # will draw 'voxelization' (spheres are active boundary nodes, cubes rigid boundary nodes)
    draw_backend='mayavi',
)

# then run with python and 3D visualization:
#   python3 -m pffdtd.sim3d.engine --sim_dir='../../sim_data/Studio/viz' --plot --draw_backend='mayavi' --json_model='model.json'
