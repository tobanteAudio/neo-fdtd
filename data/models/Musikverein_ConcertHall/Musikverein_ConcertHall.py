# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton
"""This shows a simple setup with FCC scheme, for a larger single-precision GPU run (<12GB VRAM)
"""
from pffdtd.sim3d.setup import Setup3D


class Musikverein_ConcertHall(Setup3D):
    model_file = 'model_export.json'
    mat_folder = '../../materials'
    source_index = 3
    source_signal = 'impulse'
    diff_source = True
    materials = {
        'Floor': 'mv_floor.h5',
        'Chairs': 'mv_chairs.h5',
        'Plasterboard': 'mv_plasterboard.h5',
        'Window': 'mv_window.h5',
        'Wood': 'mv_wood.h5',
    }
    duration = 2.0
    Tc = 20
    rh = 50
    fcc = True
    ppw = 5.6
    fmax = 1000.0
    save_folder = '../../sim_data/Musikverein_ConcertHall/gpu'
    save_folder_gpu = '../../sim_data/Musikverein_ConcertHall/gpu'
    compress = 3
    draw_vox = True
    draw_backend = 'polyscope'
