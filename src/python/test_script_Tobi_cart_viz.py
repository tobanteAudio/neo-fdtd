from sim_setup import sim_setup

sim_setup(
    model_json_file='../../data/models/Tobi_Office/model_export.json',
    mat_folder='../../data/materials',
    source_num=1,
    insig_type='dhann30',  # for viz
    diff_source=False,
    mat_files_dict={
        'Walls': 'ctk_walls.h5',
        'Ceiling': 'ctk_ceiling.h5',
        'Floor': 'mv_floor.h5',
    },
    duration=0.1,  # duration in seconds
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=7.5,  # for 2% phase velocity error at fmax
    fmax=500.0,
    save_folder='../../data/sim_data/Tobi/viz',  # can run python from here
    compress=0,
    draw_vox=True,
    # will draw 'voxelization' (spheres are active boundary nodes, cubes rigid boundary nodes)
    draw_backend='mayavi',
)

# then run with python and 3D visualization:
#   python3 -m fdtd.sim_fdtd --data_dir='../../data/sim_data/Tobi/viz' --plot --draw_backend='mayavi' --json_model='../../data/models/Tobi_Office/model_export.json'
