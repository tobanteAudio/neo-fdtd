# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton
"""This shows a simple setup with FCC scheme, for visualization purposes.
"""
from pffdtd.sim3d.setup import sim_setup_3d

sim_setup_3d(
    model_json_file='model_export.json',
    mat_folder='../../materials',
    source_num=3,
    insig_type='dhann30',
    diff_source=False,
    mat_files_dict={
        'Floor': 'mv_floor.h5',
        'Chairs': 'mv_chairs.h5',
        'Plasterboard': 'mv_plasterboard.h5',
        'Window': 'mv_window.h5',
        'Wood': 'mv_wood.h5',
    },
    duration=0.1,
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=5.6,  # for 2% phase velocity error at fmax
    fmax=1000.0,
    save_folder='../../sim_data/mv_fcc/viz',
    compress=0,
    draw_vox=True,
    # will draw 'voxelization' with polyscope (in which small white spheres denote rigid boundary nodes)
    draw_backend='polyscope',
)

# then run with python and 3D visualization:
#   pffdtd sim3d engine --sim_dir='../../sim_data/mv_fcc/viz' --plot --draw_backend='mayavi' --json_model='model_export.json'
