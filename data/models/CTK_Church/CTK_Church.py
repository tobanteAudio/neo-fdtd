# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton
"""This shows a simple setup with Cartesian scheme, for a single-precision GPU run (<2GB VRAM)
"""

from pffdtd.sim3d.setup import Setup3D


class CTK_Church(Setup3D):
    """Christ the King (CTK) Church
    """
    model_file = 'model_export.json'
    mat_folder = '../../materials'
    source_index = 1
    source_signal = 'impulse'
    diff_source = True
    materials = {
        'AcousticPanel': 'ctk_acoustic_panel.h5',
        'Altar': 'ctk_altar.h5',
        'Carpet': 'ctk_carpet.h5',
        'Ceiling': 'ctk_ceiling.h5',
        'Glass': 'ctk_window.h5',
        'PlushChair': 'ctk_chair.h5',
        'Tile': 'ctk_tile.h5',
        'Walls': 'ctk_walls.h5',
    }
    duration = 3.0
    Tc = 20
    rh = 50
    fcc = False
    ppw = 10.5
    fmax = 800.0
    save_folder = '../../sim_data/CTK_Church/cpu'
    save_folder_gpu = '../../sim_data/CTK_Church/gpu'
    compress = 0
    draw_vox = True
    draw_backend = 'polyscope'
