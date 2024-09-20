# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
from pathlib import Path

import numpy as np

from pffdtd.absorption.admittance import fit_to_Sabs_oct_11
from pffdtd.absorption.porous import porous_absorber
from pffdtd.geometry.math import find_third_vertex, point_along_line
from pffdtd.sim3d.model_builder import MeshModelBuilder
from pffdtd.sim3d.setup import Setup3D


class ProStudio(Setup3D):
    model_file = 'model.json'
    mat_folder = '../../sim_data/ProStudio/materials'
    source_index = 1
    source_signal = 'impulse'
    diff_source = True
    materials = {
        'ATC Left': 'wood.h5',
        'ATC Right': 'wood.h5',
        'Ceiling': 'absorber_8000_200mm_gap_200mm.h5',
        'Console': 'metal_iron.h5',
        'Couch': 'leather_arm_chair.h5',
        'Diffusor': 'wood.h5',
        'Floor': 'wood_on_concrete.h5',
        'Outboard': 'metal_iron.h5',
        'Rack': 'wood.h5',
        'Raised Floor': 'wood.h5',
        'Walls Back': 'absorber_8000_200mm_gap_200mm.h5',
        'Walls Front': 'absorber_8000_200mm_gap_100mm.h5',
        'Walls Side': 'absorber_8000_50mm.h5',
        'Windows': 'glas_thick.h5',
    }
    duration = 1.2
    Tc = 20
    rh = 50
    fcc = False
    ppw = 7.7
    fmax = 2000.0
    save_folder = '../../sim_data/ProStudio/cpu'
    save_folder_gpu = '../../sim_data/ProStudio/gpu'
    draw_vox = True
    draw_backend = 'polyscope'
    compress = 0
    rot_az_el = [0, 0]

    def generate_materials(self):
        self._print('Generate materials')
        folder = Path(self.mat_folder)

        # autopep8: off
        absorber_8000_50mm            = porous_absorber(0.05, 8000.0, 1000*(2.0**np.arange(-6, 5)), offset_zeros=True)
        absorber_8000_200mm_gap_100mm = np.array([0.10, 0.23, 0.59, 0.84, 0.91, 0.92, 0.95, 0.95, 0.94, 0.93, 0.90])
        absorber_8000_200mm_gap_200mm = np.array([0.19, 0.38, 0.69, 0.84, 0.88, 0.93, 0.95, 0.95, 0.94, 0.93, 0.90])
        glas_thick                    = np.array([0.15, 0.30, 0.27, 0.18, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02, 0.01])
        leather_arm_chair             = np.array([0.04, 0.08, 0.16, 0.20, 0.25, 0.29, 0.31, 0.29, 0.25, 0.22, 0.30])
        metal_iron                    = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.02])
        wood                          = np.array([0.10, 0.11, 0.13, 0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07, 0.07])
        wood_on_concrete              = np.array([0.01, 0.01, 0.01, 0.04, 0.04, 0.07, 0.06, 0.06, 0.07, 0.06, 0.06])

        fit_to_Sabs_oct_11(absorber_8000_50mm            , filename=folder / 'absorber_8000_50mm.h5'           )
        fit_to_Sabs_oct_11(absorber_8000_200mm_gap_100mm , filename=folder / 'absorber_8000_200mm_gap_100mm.h5')
        fit_to_Sabs_oct_11(absorber_8000_200mm_gap_200mm , filename=folder / 'absorber_8000_200mm_gap_200mm.h5')
        fit_to_Sabs_oct_11(glas_thick                    , filename=folder / 'glas_thick.h5'                   )
        fit_to_Sabs_oct_11(leather_arm_chair             , filename=folder / 'leather_arm_chair.h5'            )
        fit_to_Sabs_oct_11(metal_iron                    , filename=folder / 'metal_iron.h5'                   )
        fit_to_Sabs_oct_11(wood                          , filename=folder / 'wood.h5'                         )
        fit_to_Sabs_oct_11(wood_on_concrete              , filename=folder / 'wood_on_concrete.h5'             )
        # autopep8: on

    def generate_model(self, constants):
        self._print('Generate model')
        dir = Path('.')
        obj = dir/'obj'

        mul = 4.0 if self.fcc else 2.0
        offset = constants.h * mul

        s1 = [3.38789-3.288/2, 6.90-offset, 1.062+0.332]
        sub1 = s1.copy()
        sub1[2] = 0.0 + offset

        s2 = s1.copy()
        s2[0] += 3.288
        sub2 = s2.copy()
        sub2[2] = 0.0 + offset

        s3 = [2.25, 3.55, 0.4]

        r1 = list(find_third_vertex(s1, s2)[1])
        r1[1] += (1.0-0.0)
        r1[2] = 1.2

        r2 = r1.copy()
        r2[1] -= 1.0

        r3 = point_along_line(r2, r1, 0.20)
        r4 = point_along_line(r2, r1, 0.40)
        r5 = point_along_line(r2, r1, 0.60)
        r6 = point_along_line(r2, r1, 0.80)

        # # Couch
        # r2 = r1.copy()
        # r2[1] = 1.2
        # r2[2] = 1.3

        # r3 = with_x_offset(r2, -0.73*1.5)
        # r4 = with_x_offset(r2, -0.73*0.5)
        # r5 = with_x_offset(r2, +0.73*0.5)
        # r6 = with_x_offset(r2, +0.73*1.5)

        m = MeshModelBuilder()
        m.add('ATC Left', obj / 'atc_left.obj', [5, 5, 5], reverse=True)
        m.add('ATC Right', obj / 'atc_right.obj', [5, 5, 5], reverse=True)
        m.add('Ceiling', obj / 'ceiling.obj', [60, 60, 60])
        m.add('Console', obj / 'console.obj', [60, 60, 60], reverse=True)
        m.add('Couch', obj / 'couch.obj', [5, 5, 48], reverse=True)
        m.add('Diffusor', obj / 'diffusor.obj', [53, 33, 0], reverse=True)
        m.add('Floor', obj / 'floor.obj', [53, 33, 0])
        m.add('Outboard', obj / 'outboard.obj', [0, 0, 0], reverse=True)
        m.add('Rack', obj / 'rack.obj', [25, 25, 25], reverse=True)
        m.add('Raised Floor', obj / 'raised_floor.obj', [25, 25, 25], reverse=True)
        m.add('Walls Back', obj / 'walls_back.obj', [100, 100, 100])
        m.add('Walls Front', obj / 'walls_front.obj', [100, 100, 100])
        m.add('Walls Side', obj / 'walls_side.obj', [180, 180, 180])
        m.add('Windows', obj / 'windows.obj', [137, 207, 240], reverse=True)
        m.add_source('S1', s1)
        m.add_source('S2', s2)
        # m.add_source("S3", s3)
        # m.add_source("SUB1", sub1)
        # m.add_source("SUB2", sub2)
        m.add_receiver('R1', r1)
        m.add_receiver('R2', r2)
        m.add_receiver('R3', r3)
        m.add_receiver('R4', r4)
        m.add_receiver('R5', r5)
        m.add_receiver('R6', r6)
        m.write(self.model_file)

    def _print(self, msg):
        print(f'--PRO-STUDIO: {msg}')
