# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
from pathlib import Path

import numpy as np

from pffdtd.geometry.math import point_on_circle
from pffdtd.sim3d.model_builder import MeshModelBuilder
from pffdtd.sim3d.setup import Setup3D


class Diffusor3D(Setup3D):
    model_file = 'model.json'
    mat_folder = '../../sim_data/Diffusor3D/materials'
    source_index = 1
    source_signal = 'impulse'
    diff_source = True
    duration = 0.5
    Tc = 20
    rh = 50
    fcc = False
    ppw = 10.5
    fmax = 4000.0
    save_folder = '../../sim_data/Diffusor3D/cpu'
    save_folder_gpu = '../../sim_data/Diffusor3D/gpu'
    compress = 0
    draw_vox = True
    draw_backend = 'polyscope'
    bmin = [-3.0, -1.0, -0.1]
    bmax = [+3.0, +3.0, +2.0]

    def generate_model(self, constants):
        print('--DIFFUSOR-3D: Generate model')

        dir = Path('.')
        obj = dir/'obj'

        width = 1484.0/1000.0
        height = 1000.0/1000.0
        centre_x = width/2

        def point_at_angle(angle):
            x, y = point_on_circle((centre_x, 0), 2.0, np.deg2rad(angle))
            return [x, y, height/2]

        m = MeshModelBuilder()
        m.add('_RIGID', obj / 'diffusor.obj', [25, 25, 25], reverse=True, sides=0)
        m.add_source('S1', point_at_angle(90))
        for i, angle in enumerate(range(1, 180)):
            m.add_receiver(f'R{i}', point_at_angle(angle))

        m.write(self.model_file)
