import numpy as np

from pffdtd.sim3d.room_builder import RoomBuilder

S = 0.4
L = 7.00*S
W = 5.19*S
H = 3.70*S

room = RoomBuilder(L, W, H)
room.with_colors({
    "Ceiling": [200, 200, 200],
    "Floor": [151, 134, 122],
    "Walls": [255, 255, 255],
})

fmax = 800
ppw = 10.5
dx = 343/(fmax*ppw)
offset = dx*2

room.add_source("S1", [offset, offset, offset])
room.add_receiver("R1", [W-offset, L-offset, H-offset])

model_file = 'model.json'
room.build(model_file)
