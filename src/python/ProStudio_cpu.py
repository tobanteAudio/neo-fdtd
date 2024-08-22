from pffdtd.sim3d.sim_setup import sim_setup

sim_setup(
    model_json_file='../../data/models/ProStudio/model.json',
    mat_folder='../../data/materials',
    source_num=1,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={
        'ATC Left': 'floor_wood.h5',
        'ATC Right': 'floor_wood.h5',
        'Ceiling': 'sabine_9512.h5',
        'Console': 'sabine_01.h5',
        'Couch': 'absorber_8000_100mm.h5',
        'Floor': 'floor_wood.h5',
        'Rack': 'floor_wood.h5',
        'Raised Floor': 'floor_wood.h5',
        'Walls Back': 'absorber_8000_200mm.h5',
        'Walls Front': 'absorber_8000_200mm.h5',
        'Walls Side': 'sabine_01.h5',
    },
    duration=0.4,
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=10.5,
    fmax=800.0,
    save_folder='../../data/sim_data/ProStudio/cpu',
    save_folder_gpu='../../data/sim_data/ProStudio/gpu',
    draw_vox=True,
    draw_backend='polyscope',
    compress=0,
    rot_az_el=[0, 0],
)
