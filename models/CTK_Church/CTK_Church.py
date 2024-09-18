# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton
"""This shows a simple setup with Cartesian scheme, for a single-precision GPU run (<2GB VRAM)
"""
from pathlib import Path

import numpy as np

from pffdtd.materials.adm_funcs import fit_to_Sabs_oct_11
from pffdtd.sim3d.setup import Setup3D


class CTK_Church(Setup3D):
    """Christ the King (CTK) Church
    """
    model_file = 'model_export.json'
    mat_folder = '../../sim_data/CTK_Church/materials'
    source_index = 1
    source_signal = 'impulse'
    diff_source = True
    materials = {
        'AcousticPanel': 'acoustic_panel.h5',
        'Altar': 'altar.h5',
        'Carpet': 'carpet.h5',
        'Ceiling': 'ceiling.h5',
        'Glass': 'window.h5',
        'PlushChair': 'chair.h5',
        'Tile': 'tile.h5',
        'Walls': 'walls.h5',
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

    def generate_materials(self):
        print('--CTK-CHURCH: Generate materials')

        # autopep8: off
        acoustic_panel  = np.array([0.2  ,  0.2   , 0.42  , 0.89  , 1     , 1     , 1     , 1     , 1     , 1     , 1])
        altar           = np.array([0.25 ,  0.25  , 0.25  , 0.25  , 0.15  , 0.1   , 0.09  , 0.08  , 0.07  , 0.07  , 0.07])
        audience        = np.array([0.1  ,  0.1   , 0.1   , 0.1   , 0.07  , 0.08  , 0.1   , 0.1   , 0.11  , 0.11  , 0.11])
        carpet          = np.array([0.08 ,  0.08  , 0.08  , 0.08  , 0.24  , 0.57  , 0.69  , 0.71  , 0.73  , 0.73  , 0.73])
        ceiling         = np.array([0.19 ,  0.19  , 0.19  , 0.19  , 0.06  , 0.05  , 0.08  , 0.07  , 0.05  , 0.05  , 0.05])
        chair           = np.array([0.44 ,  0.44  , 0.44  , 0.44  , 0.56  , 0.67  , 0.74  , 0.83  , 0.87  , 0.87  , 0.87])
        tile            = np.array([0.015,  0.015 , 0.015 , 0.015 , 0.015 , 0.005 , 0.005 , 0.005 , 0.005 , 0.005 , 0.005])
        walls           = np.array([0.19 ,  0.19  , 0.19  , 0.19  , 0.06  , 0.05  , 0.08  , 0.07  , 0.05  , 0.05  , 0.05])
        window          = np.array([0.35 ,  0.35  , 0.35  , 0.35  , 0.25  , 0.18  , 0.12  , 0.07  , 0.04  , 0.04  , 0.04])

        folder = Path(self.mat_folder)
        fit_to_Sabs_oct_11(acoustic_panel , filename=folder / 'acoustic_panel.h5')
        fit_to_Sabs_oct_11(altar          , filename=folder / 'altar.h5')
        fit_to_Sabs_oct_11(audience       , filename=folder / 'audience.h5')
        fit_to_Sabs_oct_11(carpet         , filename=folder / 'carpet.h5')
        fit_to_Sabs_oct_11(ceiling        , filename=folder / 'ceiling.h5')
        fit_to_Sabs_oct_11(chair          , filename=folder / 'chair.h5')
        fit_to_Sabs_oct_11(tile           , filename=folder / 'tile.h5')
        fit_to_Sabs_oct_11(walls          , filename=folder / 'walls.h5')
        fit_to_Sabs_oct_11(window         , filename=folder / 'window.h5')
        # autopep8: on
