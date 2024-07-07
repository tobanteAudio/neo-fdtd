from sim_setup import sim_setup

sim_setup(
    model_json_file='../../data/models/Office/model_export.json',
    mat_folder='../../data/materials',
    source_num=1,
    insig_type='impulse',  # for RIR
    diff_source=True,  # for single precision
    mat_files_dict={
        'Walls': 'office_wall.h5',
        'Ceiling': 'office_ceiling.h5',
        'Floor': 'office_floor.h5',
        'Panel_1': 'office_dead.h5',
        'Panel_2': 'office_dead.h5',
        'Panel_3': 'office_dead.h5',
        'Panel_4': 'office_dead.h5',
        'Panel_5': 'office_dead.h5',
        'Panel_6': 'office_dead.h5',
        'Sofa': 'office_dead.h5',
        'Table': 'mv_wood.h5',
    },
    duration=1.0,
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=7.7,
    fmax=1000.0,
    save_folder='../../data/sim_data/Office/cpu',
    save_folder_gpu='../../data/sim_data/Office/gpu',
    compress=0,
)
# then from '../../data/sim_data/Office/gpu' folder, run (relative path for default folder structure):
#   ../../../../src/fdtd_main_gpu_single.x

# then post-process with something like:
# python -m fdtd.process_outputs --data_dir='../../data/sim_data/Office/gpu/' --fcut_lowpass 1400.0 --N_order_lowpass=8 --symmetric --fcut_lowcut 10.0 --N_order_lowcut=4 --save_wav --plot
