from sim_setup import sim_setup

sim_setup(
    model_json_file='../../data/models/Tobi_Office/model_export.json',
    mat_folder='../../data/materials',
    source_num=1,
    insig_type='impulse',  # for RIR
    diff_source=True,  # for single precision
    mat_files_dict={
        'Walls': 'tobi_wall.h5',
        'Ceiling': 'tobi_ceiling.h5',
        'Floor': 'tobi_floor.h5',
        'Panel_1': 'tobi_dead.h5',
        'Panel_2': 'tobi_dead.h5',
        'Panel_3': 'tobi_dead.h5',
        'Panel_4': 'tobi_dead.h5',
        'Panel_5': 'tobi_dead.h5',
        'Panel_6': 'tobi_dead.h5',
        'Sofa': 'tobi_dead.h5',
        'Table': 'mv_wood.h5',
    },
    duration=1.0,
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=7.7,
    fmax=1000.0,
    save_folder='../../data/sim_data/Tobi/cpu',
    save_folder_gpu='../../data/sim_data/Tobi/gpu',
    compress=0,
)
# then from '../../data/sim_data/Tobi/gpu' folder, run (relative path for default folder structure):
#   ../../../../src/fdtd_main_gpu_single.x

# then post-process with something like:
# python -m fdtd.process_outputs --data_dir='../../data/sim_data/Tobi/gpu/' --fcut_lowpass 1400.0 --N_order_lowpass=8 --symmetric --fcut_lowcut 10.0 --N_order_lowcut=4 --save_wav --plot
