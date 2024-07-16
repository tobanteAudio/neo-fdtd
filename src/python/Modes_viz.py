from sim_setup import sim_setup

# will draw 'voxelization' (spheres are active boundary nodes, cubes rigid boundary nodes)
sim_setup(
    model_json_file='../../data/models/Modes/model.json',
    mat_folder='../../data/materials',
    source_num=1,
    insig_type='dhann30',  # for viz
    diff_source=False,
    mat_files_dict={
        'Ceiling': 'concrete_painted.h5',
        'Floor': 'concrete_painted.h5',
        'Walls': 'concrete_painted.h5',
    },
    duration=0.1,
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=10.5,
    fmax=500.0,
    save_folder='../../data/sim_data/Modes/viz',  # can run python from here
    compress=0,
    draw_vox=True,
    draw_backend='mayavi',
    rot_az_el=[0, 0]
)

# then run with python and 3D visualization:
#   python3 -m sim3d.sim_fdtd --data_dir='../../data/sim_data/Modes/viz' --plot --draw_backend='mayavi' --json_model='../../data/models/Modes/model.json'
