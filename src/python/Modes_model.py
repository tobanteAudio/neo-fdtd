import numpy as np

from common.room_builder import RoomBuilder

S = 0.90
L = 7.00*S
W = 5.19*S
H = 3.70*S

room = RoomBuilder(W, L, H, wall_color=[255, 255, 255])
room.with_colors({
    "Ceiling": [200, 200, 200],
    "Floor": [151, 134, 122],
    "Walls": [255, 255, 255],
})

offset = 0.075*3

room.add_source("S1", [offset, offset, offset])
room.add_receiver("R1", [W-offset, L-offset, H-offset])

model_file = '../../data/models/Modes/model.json'
room.build(model_file)
