from sim_setup import sim_setup

sim_setup(
    model_json_file='../../data/models/Modes/model.json',
    mat_folder='../../data/materials',
    source_num=1,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={
        'Ceiling': 'concrete_painted.h5',
        'Floor': 'concrete_painted.h5',
        'Walls': 'concrete_painted.h5',
    },
    duration=17.0,
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=7.7,
    fmax=800.0,
    save_folder='../../data/sim_data/Modes/cpu',
    save_folder_gpu='../../data/sim_data/Modes/gpu',
    compress=0,
)
