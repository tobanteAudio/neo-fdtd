##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: MV_gpu.py
#
# Description: this shows a simple setup with FCC scheme, for a larger single-precision GPU run (<12GB VRAM)
#
##############################################################################
from pffdtd.sim3d.setup import sim_setup

sim_setup(
    model_json_file='model_export.json',
    draw_backend='mayavi',
    mat_folder='../../materials',
    source_num=3,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={
                    'Floor': 'mv_floor.h5',
                    'Chairs': 'mv_chairs.h5',
                    'Plasterboard': 'mv_plasterboard.h5',
                    'Window': 'mv_window.h5',
                    'Wood': 'mv_wood.h5',
    },
    duration=3.0,
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=7.7,  # for 1% phase velocity error at fmax
    fmax=2500.0,
    save_folder='../../sim_data/mv_fcc/gpu',
    save_folder_gpu='../../sim_data/mv_fcc/gpu',
    compress=3,  # apply level-3 GZIP compression to larger h5 files
)
