# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton
"""This shows a simple setup with FCC scheme, for a larger single-precision GPU run (<12GB VRAM)
"""
from pffdtd.sim3d.setup import sim_setup_3d

sim_setup_3d(
    model_json_file='model_export.json',
    mat_folder='../../materials',
    source_num=3,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={
                    'Floor': 'mv_floor.h5',
                    'Chairs': 'mv_chairs.h5',
                    'Plasterboard': 'mv_plasterboard.h5',
                    'Window': 'mv_window.h5',
                    'Wood': 'mv_wood.h5',
    },
    duration=2.0,
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=5.6,
    fmax=3200.0,
    save_folder='../../sim_data/Musikverein_ConcertHall/gpu',
    save_folder_gpu='../../sim_data/Musikverein_ConcertHall/gpu',
    compress=3,  # apply level-3 GZIP compression to larger h5 files
    draw_vox=True,
    draw_backend='polyscope',
)
