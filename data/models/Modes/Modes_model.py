from pffdtd.sim3d.constants import SimConstants
from pffdtd.sim3d.model_builder import RoomModelBuilder

S = 0.4
L = 7.00*S
W = 5.19*S
H = 3.70*S

W = 2.0
L = 3.0
H = 4.0

Tc = 20
rh = 50
fcc = False
fmax = 800
ppw = 10.5
constants = SimConstants(Tc=Tc, rh=rh, fmax=fmax,
                         PPW=ppw, fcc=fcc, verbose=False)
mul = 3.5 if fcc else 2.0
offset = constants.h * mul

room = RoomModelBuilder(L, W, H)
room.with_colors({
    "Ceiling": [200, 200, 200],
    "Floor": [151, 134, 122],
    "Walls": [255, 255, 255],
})


room.add_source("S1", [offset, offset, offset])
room.add_receiver("R1", [W-offset, L-offset, H-offset])

model_file = 'model.json'
room.build(model_file)
