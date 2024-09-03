from pffdtd.sim3d.setup import sim_setup

sim_setup(
    model_json_file='model.json',
    mat_folder='../../materials',
    source_num=1,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={
        'Ceiling': 'sabine_03.h5',
        'Floor': 'sabine_03.h5',
        'Walls': 'sabine_03.h5',
    },
    duration=3.75,
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=10.5,
    fmax=800.0,
    save_folder='../../sim_data/Modes/cpu',
    save_folder_gpu='../../sim_data/Modes/gpu',
    compress=0,
    draw_vox=False,
    draw_backend='polyscope',
    rot_az_el=[0, 0],
)
