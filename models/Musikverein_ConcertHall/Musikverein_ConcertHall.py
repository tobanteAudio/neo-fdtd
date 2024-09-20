# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton
"""This shows a simple setup with FCC scheme, for a larger single-precision GPU run (<12GB VRAM)
"""
from pathlib import Path

import numpy as np

from pffdtd.absorption.admittance import fit_to_Sabs_oct_11
from pffdtd.sim3d.setup import Setup3D


class Musikverein_ConcertHall(Setup3D):
    model_file = 'model_export.json'
    mat_folder = '../../sim_data/Musikverein_ConcertHall/materials'
    source_index = 3
    source_signal = 'impulse'
    diff_source = True
    materials = {
        'Floor': 'floor.h5',
        'Chairs': 'chairs.h5',
        'Plasterboard': 'plasterboard.h5',
        'Window': 'window.h5',
        'Wood': 'wood.h5',
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

    def generate_materials(self):
        print('--CONCERT-HALL: Generate materials')
        folder = Path(self.mat_folder)

        # autopep8: off
        chairs       = np.array([0.22, 0.22, 0.22, 0.22, 0.26, 0.3 , 0.33, 0.34, 0.34, 0.34, 0.34])
        floor        = np.array([0.14, 0.14, 0.14, 0.14, 0.1 , 0.06, 0.08, 0.1 , 0.1 , 0.1 , 0.1])
        plasterboard = np.array([0.15, 0.15, 0.15, 0.15, 0.1 , 0.06, 0.04, 0.04, 0.05, 0.05, 0.05])
        window       = np.array([0.35, 0.35, 0.35, 0.35, 0.25, 0.18, 0.12, 0.07, 0.04, 0.04, 0.04])
        wood         = np.array([0.25, 0.25, 0.25, 0.25, 0.15, 0.1 , 0.09, 0.08, 0.07, 0.07, 0.07])
        fit_to_Sabs_oct_11(chairs       , filename=folder / 'chairs.h5')
        fit_to_Sabs_oct_11(floor        , filename=folder / 'floor.h5')
        fit_to_Sabs_oct_11(plasterboard , filename=folder / 'plasterboard.h5')
        fit_to_Sabs_oct_11(window       , filename=folder / 'window.h5')
        fit_to_Sabs_oct_11(wood         , filename=folder / 'wood.h5')
        # autopep8: on
