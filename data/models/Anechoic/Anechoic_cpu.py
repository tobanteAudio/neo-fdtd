from pffdtd.sim3d.setup import sim_setup

sim_setup(
    model_json_file='model.json',
    mat_folder='../../materials',
    source_num=1,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={
        # 'Floor': 'floor_wood.h5',
        'Walls': 'concrete_painted.h5',
    },
    duration=0.5,
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=12.0,
    fmax=1000.0,
    save_folder='../../sim_data/Anechoic/cpu',
    save_folder_gpu='../../sim_data/Anechoic/gpu',
    draw_vox=True,
    draw_backend='polyscope',
    compress=0,
    rot_az_el=[0, 0],
    bmax=[18.0, 2.0, 18.0],
    bmin=[0, 0, 0],
)
