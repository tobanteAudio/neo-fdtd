from sim_setup import sim_setup

sim_setup(
    model_json_file='../../data/models/Studio/model.json',
    mat_folder='../../data/materials',
    source_num=1,
    insig_type='impulse',  # for RIR
    diff_source=True,  # for single precision
    mat_files_dict={
        'Walls': 'office_wall.h5',
        'Ceiling': 'office_ceiling.h5',
        'Floor': 'office_floor.h5',
        'Panel': 'office_dead.h5',
        'Sofa': 'office_dead.h5',
        'Table': 'mv_wood.h5',
    },
    duration=1.0,
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=7.7,
    fmax=1000.0,
    save_folder='../../data/sim_data/Studio/cpu',
    save_folder_gpu='../../data/sim_data/Studio/gpu',
    compress=0,
)
