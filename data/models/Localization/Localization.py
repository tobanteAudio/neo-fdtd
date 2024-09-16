# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import numpy as np

from pffdtd.sim3d.model_builder import RoomModelBuilder
from pffdtd.sim3d.setup import Setup3D


class Localization(Setup3D):
    model_json_file = 'model.json'
    mat_folder = '../../materials'
    source_index = 1
    source_signal = 'impulse'
    diff_source = True
    materials = {
        'Ceiling': 'sabine_9512.h5',
        'Floor': 'sabine_9512.h5',
        'Walls': 'sabine_9512.h5',
    }
    duration = 0.4
    Tc = 20
    rh = 50
    fcc_flag = False
    ppw = 10.5
    fmax = 800.0
    save_folder = '../../sim_data/Localization/cpu'
    save_folder_gpu = '../../sim_data/Localization/gpu'
    compress = 0
    draw_vox = True
    draw_backend = 'polyscope'

    def generate_model(self, constants):
        L = 3.0
        W = 3.0
        H = 3.0

        source = [W/2, L-0.1, H/2]
        mics = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0.5, np.sqrt(3)/2, 0]),
            np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/3]),
        ]

        room = RoomModelBuilder(L, W, H)
        room.with_colors({
            "Ceiling": [200, 200, 200],
            "Floor": [151, 134, 122],
            "Walls": [255, 255, 255],
        })

        room.add_source("S1", source)
        room.add_receiver("R1", list(mics[1-1]/2+[0.5, 0.5, 0.5]))
        room.add_receiver("R2", list(mics[2-1]/2+[0.5, 0.5, 0.5]))
        room.add_receiver("R3", list(mics[3-1]/2+[0.5, 0.5, 0.5]))
        room.add_receiver("R4", list(mics[4-1]/2+[0.5, 0.5, 0.5]))
        room.build(self.model_file)
