# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton
"""This shows a simple setup with Cartesian scheme, for a single-precision GPU run (<2GB VRAM)
"""

from pffdtd.sim3d.setup import sim_setup_3d

sim_setup_3d(
    model_json_file='model_export.json',
    mat_folder='../../materials',
    source_num=1,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={
        'AcousticPanel': 'ctk_acoustic_panel.h5',
        'Altar': 'ctk_altar.h5',
        'Carpet': 'ctk_carpet.h5',
        'Ceiling': 'ctk_ceiling.h5',
        'Glass': 'ctk_window.h5',
        'PlushChair': 'ctk_chair.h5',
        'Tile': 'ctk_tile.h5',
        'Walls': 'ctk_walls.h5',
    },
    duration=3.0,
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=10.5,  # for 1% phase velocity error at fmax
    fmax=1400.0,
    save_folder='../../sim_data/ctk_cart/cpu',
    save_folder_gpu='../../sim_data/ctk_cart/gpu',
    compress=0,
    draw_vox=True,
    draw_backend='polyscope',
)
