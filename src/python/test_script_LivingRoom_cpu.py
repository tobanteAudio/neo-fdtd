from sim_setup import sim_setup
from common.room_builder import RoomBuilder

L = 6.0
W = 3.65
H = 3.12
model_file = '../../data/models/LivingRoom/model.json'


builder = RoomBuilder(W, L, H, wall_color=[255, 255, 255])
builder.with_colors({
    "Ceiling": [200, 200, 200],
    "Floor": [151, 134, 122],
    "Table": [130, 75, 25],
    "TV_Stand": [80, 80, 80],
    "Sofa": [25, 25, 25],
    "Walls": [255, 255, 255],
})

builder.add_box("Sofa", [0.98, 2.52, 0.48], [0.05, L-2.52-0.05, 0.06])
builder.add_box("Table", [0.75, 1.25, 0.02], [0.05, L-2.52-0.1-1.25, 0.66])
builder.add_box("Table", [0.8, 1.8, 0.02], [W-0.8-0.05, 1, 0.7])
builder.add_box("Table", [0.8, 0.8, 0.02], [W/2-0.4, L-1-0.4, 0.3])
builder.add_box("TV_Stand", [0.5, 1.6, 0.35], [W-0.5-0.05, L-2, 0.03])

builder.add_source("PC Speaker Left", [0.2, L-2.52-1.25+0.1, 0.8])
builder.add_source("PC Speaker Right", [0.2, L-2.52-0.2, 0.8])
builder.add_receiver("PC Listener", [1.0, L-2.52-1.4/2, 1.2])

builder.add_receiver("TV Listener", [2.0, L-2.52-1.4/2, 1.2])

builder.build(model_file)


sim_setup(
    model_json_file=model_file,
    mat_folder='../../data/materials',
    source_num=1,
    insig_type='impulse',  # for RIR
    diff_source=True,  # for single precision
    mat_files_dict={
        'Ceiling': 'office_ceiling.h5',
        'Floor': 'office_floor.h5',
        'Table': 'mv_wood.h5',
        'TV_Stand': 'mv_wood.h5',
        'Sofa': 'office_dead.h5',
        'Walls': 'office_wall.h5',
    },
    duration=1.5,
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=7.7,
    fmax=1000.0,
    save_folder='../../data/sim_data/LivingRoom/cpu',
    save_folder_gpu='../../data/sim_data/LivingRoom/gpu',
    compress=0,
)
