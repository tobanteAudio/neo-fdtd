# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.sim3d import process_outputs
from pffdtd.sim3d import room_geometry
from pffdtd.sim3d import setup


@click.group(help='3D wave-equation.')
def sim3d():
    pass


sim3d.add_command(process_outputs.main)
sim3d.add_command(room_geometry.main)
sim3d.add_command(setup.main)
