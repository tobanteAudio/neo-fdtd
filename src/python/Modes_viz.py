from sim_setup import sim_setup

sim_setup(
    model_json_file='../../data/models/Modes/model.json',
    mat_folder='../../data/materials',
    source_num=1,
    insig_type='dhann30',  # for viz
    diff_source=False,
    mat_files_dict={
        'Ceiling': 'absorber_8000_200mm.h5',
        'Floor': 'absorber_8000_200mm.h5',
        'Walls': 'absorber_8000_200mm.h5',
    },
    duration=0.1,  # duration in seconds
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=7.7,  # for 2% phase velocity error at fmax
    fmax=500.0,
    save_folder='../../data/sim_data/Modes/viz',  # can run python from here
    compress=0,
    draw_vox=True,
    # will draw 'voxelization' (spheres are active boundary nodes, cubes rigid boundary nodes)
    draw_backend='mayavi',
)

# then run with python and 3D visualization:
#   python3 -m sim3d.sim_fdtd --data_dir='../../data/sim_data/Modes/viz' --plot --draw_backend='mayavi' --json_model='../../data/models/Modes/model.json'
