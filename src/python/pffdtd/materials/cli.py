# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click
import numpy as np
import pathlib

from pffdtd.materials.adm_funcs import plot_DEF_admittance, read_mat_DEF


@click.group(help='Materials.')
def materials():
    pass


@materials.command(help='Plot material.')
@click.argument('material_file', nargs=1, type=click.Path(exists=True))
def plot(material_file):
    frequencies = np.logspace(np.log10(10), np.log10(20e3), 4000)
    material = read_mat_DEF(pathlib.Path(material_file))
    plot_DEF_admittance(frequencies, material)


# freq-independent impedance from reflection coefficients
# write_freq_ind_mat_from_Yn(convert_R_to_Yn(0.90),filename=Path(write_folder / 'R90_mat.h5'))
# write_freq_ind_mat_from_Yn(convert_R_to_Yn(0.5),filename=Path(write_folder / 'R50.h5'))

# #input DEF values directly
# write_freq_dep_mat(npa([[0,1.0,0],[2,3,4]]),filename=Path(write_folder / 'ex_mat.h5'))
