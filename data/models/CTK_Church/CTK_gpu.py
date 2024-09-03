##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: CTK_gpu.py
#
# Description: this shows a simple setup with Cartesian scheme, for a single-precision GPU run (<2GB VRAM)
#
##############################################################################

from pffdtd.sim3d.setup import sim_setup_3d

sim_setup_3d(
    model_json_file='model_export.json',
    mat_folder='../../materials',
    source_num=1,
    insig_type='impulse',  # for RIR
    diff_source=True,  # for single precision
    mat_files_dict={
                    'AcousticPanel': 'ctk_acoustic_panel.h5',
                    'Altar': 'ctk_altar.h5',
                    'Carpet': 'ctk_carpet.h5',
                    'Ceiling': 'ctk_ceiling.h5',
                    'Glass': 'ctk_window.h5',
                    'PlushChair': 'ctk_chair.h5',
                    'Tile': 'ctk_tile.h5',
                    'Walls': 'ctk_walls.h5',
    },
    duration=3.0,  # duration in seconds
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=10.5,  # for 1% phase velocity error at fmax
    fmax=1400.0,
    save_folder='../../sim_data/ctk_cart/gpu',
    save_folder_gpu='../../sim_data/ctk_cart/gpu',
    compress=0,
)
