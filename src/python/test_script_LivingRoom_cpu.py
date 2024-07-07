from sim_setup import sim_setup

sim_setup(
    model_json_file='../../data/models/LivingRoom/model.json',
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
