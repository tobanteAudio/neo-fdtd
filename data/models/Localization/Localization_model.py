import numpy as np

from pffdtd.sim3d.room_builder import RoomBuilder

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

builder = RoomBuilder(L, W, H)
builder.with_colors({
    "Ceiling": [200, 200, 200],
    "Floor": [151, 134, 122],
    "Walls": [255, 255, 255],
})

builder.add_source("S1", source)
builder.add_receiver("R1", list(mics[1-1]/2+[0.5, 0.5, 0.5]))
builder.add_receiver("R2", list(mics[2-1]/2+[0.5, 0.5, 0.5]))
builder.add_receiver("R3", list(mics[3-1]/2+[0.5, 0.5, 0.5]))
builder.add_receiver("R4", list(mics[4-1]/2+[0.5, 0.5, 0.5]))
builder.build('model.json')
