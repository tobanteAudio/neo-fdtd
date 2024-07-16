from sim_setup import sim_setup

sim_setup(
    model_json_file='../../data/models/Modes/model.json',
    mat_folder='../../data/materials',
    source_num=1,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={
        'Ceiling': 'almost_rigid.h5',
        'Floor': 'almost_rigid.h5',
        'Walls': 'almost_rigid.h5',
    },
    duration=7.5,
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=10.5,
    fmax=400.0,
    save_folder='../../data/sim_data/Modes/cpu',
    save_folder_gpu='../../data/sim_data/Modes/gpu',
    compress=0,
    rot_az_el=[0, 0],
)
