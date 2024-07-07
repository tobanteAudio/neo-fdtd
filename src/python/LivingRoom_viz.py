from sim_setup import sim_setup

sim_setup(
    model_json_file='../../data/models/LivingRoom/model.json',
    mat_folder='../../data/materials',
    source_num=1,
    insig_type='dhann30',  # for viz
    diff_source=False,
    mat_files_dict={
        'Ceiling': 'wall_concrete_painted.h5',
        'Floor': 'floor_wood.h5',
        'Table': 'mv_wood.h5',
        'TV_Stand': 'mv_wood.h5',
        'Sofa': 'absorber_8000_100mm.h5',
        'Walls': 'wall_concrete_painted.h5'
    },
    duration=0.1,
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=7.7,  # for 2% phase velocity error at fmax
    fmax=500.0,
    save_folder='../../data/sim_data/LivingRoom/viz',  # can run python from here
    compress=0,
    draw_vox=True,
    # will draw 'voxelization' (spheres are active boundary nodes, cubes rigid boundary nodes)
    draw_backend='mayavi',
)

# then run with python and 3D visualization:
#   python3 -m fdtd.sim_fdtd --data_dir='../../data/sim_data/LivingRoom/viz' --plot --draw_backend='mayavi' --json_model='../../data/models/LivingRoom/model.json'
