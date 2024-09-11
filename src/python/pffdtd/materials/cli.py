# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click
import numpy as np
import pathlib

from pffdtd.materials.adm_funcs import plot_DEF_admittance, read_mat_DEF
from pffdtd.materials.build import build


@click.group(help="Materials.")
def materials():
    pass


@materials.command(help="Plot material.")
@click.argument('material_file', nargs=1, type=click.Path(exists=True))
def plot(material_file):
    frequencies = np.logspace(np.log10(10), np.log10(20e3), 4000)
    material = read_mat_DEF(pathlib.Path(material_file))
    plot_DEF_admittance(frequencies, material)


materials.add_command(build)
