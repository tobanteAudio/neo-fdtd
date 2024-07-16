from sim_setup import sim_setup

sim_setup(
    model_json_file='../../data/models/Studio/model.json',
    mat_folder='../../data/materials',
    source_num=1,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={
        'Absorber M': 'absorber_8000_100mm.h5',
        'Absorber L': 'absorber_8000_200mm.h5',
        'Ceiling': 'concrete_painted.h5',
        'Diffusor': 'door_wood.h5',
        'Floor': 'floor_wood.h5',
        'Sofa': 'absorber_8000_100mm.h5',
        'Speaker_Cabinet': 'door_wood.h5',
        'Table': 'door_wood.h5',
        'Walls': 'concrete_painted.h5',
    },
    duration=1.5,
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=10.5,
    fmax=800.0,
    save_folder='../../data/sim_data/Studio/cpu',
    save_folder_gpu='../../data/sim_data/Studio/gpu',
    compress=0,
)
