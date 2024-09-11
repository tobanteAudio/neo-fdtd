# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

from pffdtd.geometry.math import find_third_vertex
from pffdtd.sim3d.model_builder import RoomModelBuilder

S = 0.90
L = 7.00*S
W = 5.19*S
H = 3.70*S

src_height = 1.2
src_backwall = 1
src_distance = 2.2
src_left = [W/2-src_distance/2, L-src_backwall, src_height]
src_right = [W/2+src_distance/2, L-src_backwall, src_height]

l1, l2 = find_third_vertex(src_left, src_right)
listener = l1 if l1[1] < l2[1] else l2

p1 = listener.copy()
p1[2] = 0.9

p2 = p1.copy()
p2[2] = 1.5

p3 = p1.copy()
p3[2] = 2.0

room = RoomModelBuilder(L, W, H)
room.with_colors({
    "Absorber M": [111, 55, 10],
    "Absorber L": [111, 55, 10],
    "Ceiling": [200, 200, 200],
    "Diffusor": [140, 80, 35],
    "Floor": [151, 134, 122],
    "Table": [130, 75, 25],
    "Sofa": [25, 25, 25],
    "Speaker_Cabinet": [15, 15, 15],
    "Walls": [255, 255, 255],
})

room.add_box("Absorber M", [0.1, 2.5, 1.5], [0.2, L-2.5-0.75, 0.75])
room.add_box("Absorber M", [0.1, 2.5, 1.5], [W-0.1-0.2, L-2.5-0.75, 0.75])

room.add_box("Absorber L", [1.0, 0.2, 2.5], [src_left[0]-1.0/2, L-0.2-0.1, 0.25])
room.add_box("Absorber L", [1.0, 0.2, 2.5], [src_right[0]-1.0/2, L-0.2-0.1, 0.25])

room.add_box("Absorber L", [2.5, 2.0, 0.2], [W/2-2.5/2, L-2-0.5, H-0.3])
room.add_box("Absorber L", [2.5, 2.0, 0.2], [W/2-2.5/2, L-2-2.5-0.5, H-0.3])

room.add_diffusor_1d([2, 15*0.0254, 1.5], [W/2-1, 0.05, 0.7], 3*0.0254)
room.add_box("Absorber L", [1.0, 0.2, 2.5], [0.1, 0.1, 0.25])
room.add_box("Absorber L", [1.0, 0.2, 2.5], [W-1.0-0.1, 0.1, 0.25])

room.add_box("Sofa", [2.52, 0.98, 0.48], [W/2-2.52/2, 0.4, 0.05])
room.add_box("Table", [1.8, 0.8, 0.02], [W/2-1.8/2, listener[1]+0.4, 0.7])

speaker_box = [0.435, 0.490, 0.650]
speaker_mid = [0.217, -0.075, 0.520]
room.add_cabinet_speaker("Speaker Left", src_left, speaker_box, speaker_mid)
room.add_cabinet_speaker("Speaker Right", src_right, speaker_box, speaker_mid)
# room.add_cabinet_speaker("Speaker Left", src_left)
# room.add_cabinet_speaker("Speaker Right", src_right)
# room.add_source("Speaker Left", src_left)
# room.add_source("Speaker Right", src_right)

room.add_receiver("Engineer", listener.tolist())
room.add_receiver("P1", p1.tolist())
room.add_receiver("P2", p2.tolist())
room.add_receiver("P3", p3.tolist())

model_file = 'model.json'
room.build(model_file)
