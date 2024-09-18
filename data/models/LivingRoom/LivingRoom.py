# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import pathlib

from pffdtd.sim3d.model_builder import MeshModelBuilder
from pffdtd.sim3d.setup import Setup3D


class LivingRoom(Setup3D):
    model_file = 'model.json'
    mat_folder = '../../materials'
    source_index = 1
    source_signal = 'impulse'
    diff_source = True
    materials = {
        'Book Shelf': 'floor_wood.h5',
        'Ceiling': 'concrete_painted.h5',
        'Coffee Table': 'floor_wood.h5',
        'Couch': 'absorber_8000_100mm.h5',
        'Desk': 'floor_wood.h5',
        'Door': 'floor_wood.h5',
        'Floor': 'floor_wood.h5',
        'Kallax': 'floor_wood.h5',
        'Monitors': 'floor_wood.h5',
        'Speakers': 'floor_wood.h5',
        'Speaker Stands': 'door_iron.h5',
        'Table': 'floor_wood.h5',
        'TV 42': 'floor_wood.h5',
        'TV 55': 'floor_wood.h5',
        'TV Table': 'floor_wood.h5',
        'Walls': 'concrete_painted.h5',
        'Window': 'glas_thick.h5',
    }
    duration = 2.0
    Tc = 20
    rh = 50
    fcc = False
    ppw = 10.5
    fmax = 800.0
    save_folder = '../../sim_data/LivingRoom/cpu'
    save_folder_gpu = '../../sim_data/LivingRoom/gpu'
    compress = 0
    draw_vox = True
    draw_backend = 'polyscope'

    def generate_model(self, constants):
        dir = pathlib.Path('.')
        obj = dir/'obj'

        s1 = [3.65-0.5, 6.0-0.3, 1.2]
        s2 = [3.65-0.5, 6.0-2.5, 1.2]

        r1 = [0.6, 6.0-1.3, 1.1]
        r2 = [0.6, 6.0-2.0, 1.1]

        m = MeshModelBuilder()
        m.add('Book Shelf', obj / 'book_shelf.obj', [200, 200, 200], reverse=True)
        m.add('Ceiling', obj / 'ceiling.obj', [150, 150, 150], reverse=True)
        m.add('Coffee Table', obj / 'coffee_table.obj', [10, 10, 10], reverse=True)
        m.add('Couch', obj / 'couch.obj', [29, 50, 112], reverse=True)
        m.add('Desk', obj / 'desk.obj', [103, 70, 55], reverse=True)
        m.add('Door', obj / 'door.obj', [103, 70, 55], reverse=True)
        m.add('Floor', obj / 'floor.obj', [133, 94, 66], reverse=True)
        m.add('Kallax', obj / 'kallax.obj', [200, 200, 200], reverse=True)
        m.add('Monitors', obj / 'monitors.obj', [15, 15, 15], reverse=True)
        m.add('Speakers', obj / 'speakers.obj', [25, 25, 25], reverse=True)
        m.add('Speaker Stands', obj / 'speaker_stands.obj', [5, 5, 5], reverse=True)
        m.add('Table', obj / 'table.obj', [200, 200, 200], reverse=True)
        m.add('TV 42', obj / 'tv_42.obj', [10, 10, 10], reverse=True)
        m.add('TV 55', obj / 'tv_55.obj', [10, 10, 10], reverse=True)
        m.add('TV Table', obj / 'tv_table.obj', [120, 120, 120], reverse=True)
        m.add('Walls', obj / 'walls.obj', [175, 175, 175], reverse=True)
        m.add('Window', obj / 'window.obj', [137, 207, 240], reverse=True)
        m.add_source('S1', s1)
        m.add_source('S2', s2)
        m.add_receiver('R1', r1)
        m.add_receiver('R2', r2)
        m.write(self.model_file)
